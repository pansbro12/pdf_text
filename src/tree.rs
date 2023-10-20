use pdf_render::TextSpan;
use pathfinder_geometry::{
    vector::Vector2F,
    rect::RectF,
};
#[cfg(feature = "ocr")]
use tesseract_plumbing::Text;

use std::collections::BTreeSet;
use std::iter::once;
use std::sync::Arc;
use itertools::{Itertools};
use ordered_float::NotNan;
use crate::entry::{Flow, Line, Run, RunType, Word};
use crate::util::{is_number, avg, CellContent};
use crate::text::{concat_text};
use std::mem::take;
use colored::Colorize;
use table::Table;

pub fn build(spans: &[TextSpan], bbox: RectF, lines: &[[f32; 4]]) -> Node {
    if spans.len() == 0 {
        return Node::singleton(&[]);
    }
    // dbg!(&bbox.0);

    let span_text: Vec<&str> = spans.iter().map(|span| span.text.as_str()).collect();


    let mut boxes: Vec<(RectF, usize)> = spans.iter().enumerate().map(|(i, t)| (t.rect, i)).collect();
    let mut boxes = boxes.as_mut_slice();

    let avg_font_size = avg(spans.iter().map(|s| s.font_size)).unwrap();
    //checks class of box if header or num true, otherwise checkes the average font size of the entire text and checks for greater than
    let probably_header = |boxes: &[(RectF, usize)]| {
        let class = classify(boxes.iter().filter_map(|&(_, i)| spans.get(i)));
        if matches!(class, Class::Header | Class::Number) {
            // if matches!(class, Class::Header ) {
            return true;
        }
        let f = avg(boxes.iter().filter_map(|&(_, i)| spans.get(i)).map(|s| s.font_size)).unwrap();
        f > avg_font_size
    };
    let probably_footer = |boxes: &mut [(RectF, usize)]| {
        sort_x(boxes);
        let x_gaps: Vec<f32> = gaps(avg_font_size, boxes, |r| (r.min_x(), r.max_x())).map(|(a, b)| b).collect();
        // let f = avg(boxes.iter().filter_map(|&(_, i)| spans.get(i)).map(|s| s.font)).unwrap();


        let split_cells = split_by(boxes, &x_gaps, |r| r.min_x());
        let count = split_cells.count();
        // let count = split_cells.map(|each| {
        //     dbg!(&spans[each[0].1]);
        //     each
        // }).filter(|cell| probably_header(cell)).count();
        // dbg!(count);
        count == x_gaps.len() + 1
    };
//these sections sort the box values, find whether the first and last values are outside of a middle boundary and then uses their indexes to
    //sorting by y or x will basically shift everything
    sort_y(boxes);
    let (top, bottom) = top_bottom_gap(boxes, bbox);
    if let Some(bottom) = bottom {
        if probably_footer(&mut boxes[bottom..]) {
            boxes = &mut boxes[..bottom];
        }
    }
    if let Some(top) = top {
        if probably_header(&mut boxes[..top]) {
            // boxes = &mut boxes[top..];
        }
    }
    sort_x(boxes);
    let (left, right) = left_right_gap(boxes, bbox);
    if let Some(right) = right {
        if probably_header(&boxes[right..]) {
            boxes = &mut boxes[..right];
        }
    }
    if let Some(left) = left {
        if probably_header(&boxes[..left]) {
            // boxes = &mut boxes[left..];
        }
    }

    let lines = analyze_lines(lines);
    println!("{}", "initial split".cyan());
    sort_occurence(boxes);
    let node = split(boxes, &spans, &lines);
    // dbg!(&node);
    node
}


///
fn analyze_lines(lines: &[[f32; 4]]) -> Lines {
    let mut hlines = BTreeSet::new();
    let mut vlines = BTreeSet::new();

    for &[x1, y1, x2, y2] in lines {
        if x1 == x2 {
            vlines.insert(NotNan::new(x1).unwrap());
        } else if y1 == y2 {
            hlines.insert(NotNan::new(y1).unwrap());
        }
    }

    fn dedup(lines: impl Iterator<Item=NotNan<f32>>) -> Vec<(f32, f32)> {
        let threshold = 10.0; //another undescribed value
        let mut out = vec![];
        let mut lines = lines.map(|f| *f).peekable();
        while let Some(start) = lines.next() {
            let mut last = start;
            while let Some(&p) = lines.peek() {
                if last + threshold > p {
                    last = p;
                    lines.next();
                } else {
                    break;
                }
            }
            out.push((start, last));
        }
        out
    }

    let hlines = dedup(hlines.iter().cloned());
    let vlines = dedup(vlines.iter().cloned());

    let mut line_grid = vec![false; vlines.len() * hlines.len()];
    for &[x1, y1, x2, y2] in lines {
        if x1 == x2 {
            let v_idx = vlines.iter().position(|&(a, b)| a <= x1 && x1 <= b).unwrap_or(vlines.len());
            let h_start = hlines.iter().position(|&(a, _b)| y1 >= a).unwrap_or(hlines.len());
            let h_end = hlines.iter().position(|&(_a, b)| y2 <= b).unwrap_or(hlines.len());
            for h in h_start..h_end {
                line_grid[v_idx * hlines.len() + h] = true;
            }
        } else if y1 == y2 {
            let h_idx = hlines.iter().position(|&(a, b)| a <= y1 && y1 <= b).unwrap_or(hlines.len());
            let v_start = vlines.iter().position(|&(a, _b)| x1 >= a).unwrap_or(vlines.len());
            let v_end = vlines.iter().position(|&(_a, b)| x2 <= b).unwrap_or(vlines.len());
            for v in v_start..v_end {
                line_grid[v * hlines.len() + h_idx] = true;
            }
        }
    }


    //println!("hlines: {:?}", hlines);
    //println!("vlines: {:?}", vlines);

    Lines { hlines, vlines, line_grid }
}

pub struct Lines {
    hlines: Vec<(f32, f32)>,
    vlines: Vec<(f32, f32)>,
    line_grid: Vec<bool>,
}

#[derive(Copy, Clone, Debug)]
struct Span {
    start: NotNan<f32>,
    end: NotNan<f32>,
}

impl Span {
    ///return a touple containing the left and right x values of the of a rect as a span
    fn horiz(rect: &RectF) -> Option<Self> {
        Self::new(rect.min_x(), rect.max_x())
    }
    ///return a touple containing the top and the bottom y values  of a rect as a span
    fn vert(rect: &RectF) -> Option<Self> {
        Self::new(rect.min_y(), rect.max_y())
    }

    ///returns a new span, corrects inputs for positive linearity along axis
    fn new(mut start: f32, mut end: f32) -> Option<Self> {
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }
        Some(Span {
            start: NotNan::new(start).ok()?,
            end: NotNan::new(end).ok()?,
        })
    }

    ///returns the span produced by the intersection of the two spans.
    fn intersect(self, other: Span) -> Option<Span> {
        if self.start <= other.end && other.start <= self.end {
            Some(Span {
                start: self.start.max(other.start),
                end: self.end.min(other.end),
            })
        } else {
            None
        }
    }
    ///for spans that overlap such that the the start of the calling span and the end of the called
    /// span are not overlapped by the other span
    fn union(self, other: Span) -> Option<Span> {
        if self.start <= other.end && other.start <= self.end {
            Some(Span {
                start: self.start.min(other.start),
                end: self.end.max(other.end),
            })
        } else {
            None
        }
    }
}


///called when there is splits along the x and y axis
pub fn split2(boxes: &mut [(RectF, usize)], spans: &[TextSpan], lines_info: &Lines) -> Node {
    use std::mem::replace;

    #[derive(Debug)]
    enum LineTag {
        Unknown,
        Text,
        Table,
    }

    sort_y(boxes);
    let mut lines = vec![];
    let mut y = Span::vert(&boxes[0].0).unwrap();
    let _items = vec![boxes[0]];
    //joins contiguous rects into lines represented by the span that they occupy and the items that are contained
    let build_line = |boxes: &[(RectF, usize)]| -> (LineTag, Span, Vec<(Span, Vec<usize>)>) {
        //
        let mut line = vec![];
        let mut x = Span::horiz(&boxes[0].0).unwrap();
        let mut y = Span::vert(&boxes[0].0).unwrap();
        let mut items = vec![boxes[0].1];

        //if rects overlap or are contiguous in space with the previous one then then add the i
        // value to items and join the rect to the previous one
        // if not then the previous line to lines and replace the items vector associated with the line
        // with a new vector containing the i value for the start of the new line.

        for &(rect, i) in &boxes[1..] {
            y = y.union(Span::vert(&rect).unwrap()).unwrap();
            let x2 = Span::horiz(&rect).unwrap();
            if let Some(u) = x.union(x2) {
                x = u;
                items.push(i);
            } else {
                line.push((x, replace(&mut items, vec![i])));
                x = x2;
            }
        }
        line.push((x, items));

        let f = avg(boxes.iter().filter_map(|&(_, i)| spans.get(i)).map(|s| s.font_size)).unwrap();

        let max_gap = line.iter().tuple_windows().map(|(l, r)| r.0.start - l.0.end).max();
        let tag = match max_gap {
            None => LineTag::Unknown,
            Some(x) if x.into_inner() < 0.3 * f => LineTag::Text,// another unexplained hard coded value seems like its just the ratio of font size
            Some(_) => LineTag::Table,
        };

        (tag, y, line)
    };

    let mut line = vec![boxes[0]];
    for &(rect, i) in &boxes[1..] {
        let y2 = Span::vert(&rect).unwrap();
        if let Some(overlap) = y.intersect(y2) {
            y = overlap;
        } else {
            sort_x(&mut line);
            lines.push(build_line(&line));
            line.clear();
            y = y2
        }
        line.push((rect, i));
    }
    sort_x(&mut line);
    lines.push(build_line(&line));


    let mut vparts = vec![];
    let mut start = 0;
    while let Some(p) = lines[start..].iter().position(|(tag, _, _line)| matches!(tag, LineTag::Unknown | LineTag::Table)) {
        let table_start = start + p;
        let table_end = lines[table_start + 1..].iter().position(|(tag, _, _)| matches!(tag, LineTag::Text)).map(|e| table_start + 1 + e).unwrap_or(lines.len());

        for &(_, y, ref line) in &lines[start..table_start] {
            vparts.push((y, Node::Final { indices: line.iter().flat_map(|(_, indices)| indices.iter().cloned()).collect() }));
        }

        let lines = &lines[table_start..table_end];
        start = table_end;

        let mut columns: Vec<Span> = vec![];
        for (_, _, line) in lines.iter() {
            for &(x, ref _parts) in line.iter() {
                // find any column that is contained in this
                let mut found = 0;
                for span in columns.iter_mut() {
                    if let Some(overlap) = span.intersect(x) {
                        *span = overlap;
                        found += 1;
                    }
                }
                if found == 0 {
                    columns.push(x);
                }
            }
        }
        let avg_vgap = avg(lines.iter().map(|(_, y, _)| y).tuple_windows().map(|(a, b)| *(b.start - a.end)));

        columns.sort_by_key(|s| s.start);

        let _buf = String::new();

        let d_threshold = avg_vgap.unwrap_or(0.0);
        let mut prev_end = None;

        let mut table: Table<Vec<usize>> = Table::empty(lines.len() as u32, columns.len() as u32);

        let mut row = 0;
        for (_, span, line) in lines {
            let mut col = 0;

            let combine = prev_end.map(|y: NotNan<f32>| {
                if *(span.start - y) < d_threshold {
                    !lines_info.hlines.iter().map(|(a, b)| 0.5 * (a + b)).any(|l| *y < l && *span.start > l)
                } else {
                    false
                }
            }).unwrap_or(false);

            if !combine {
                row += 1;
            }

            for &(x, ref parts) in line {
                let mut cols = columns.iter().enumerate()
                    .filter(|&(_, &x2)| x.intersect(x2).is_some())
                    .map(|(i, _)| i);

                let first_col = cols.next().unwrap();
                let last_col = cols.last().unwrap_or(first_col);

                if let Some(cell) = combine.then(|| table.get_cell_value_mut(row, first_col as u32)).flatten() {
                    // append to previous line
                    cell.extend_from_slice(parts);
                } else {
                    let colspan = (last_col - first_col) as u32 + 1;
                    let rowspan = 1;
                    table.set_cell(parts.clone(), row, first_col as u32, rowspan, colspan);
                }
                col = last_col + 1;
            }
            prev_end = Some(span.end);
        }
        let y = Span { start: lines[0].1.start, end: lines.last().unwrap().1.end };
        vparts.push((y, Node::Table { table }));
    }
    for &(_, y, ref line) in &lines[start..] {
        vparts.push((y, Node::Final { indices: line.iter().flat_map(|(_, indices)| indices.iter().cloned()).collect() }));
    }

    if vparts.len() > 1 {
        let y = vparts.iter().tuple_windows().map(|(a, b)| 0.5 * (a.0.end + b.0.start).into_inner()).collect();
        Node::Grid {
            tag: NodeTag::Complex,
            x: vec![],
            y,
            cells: vparts.into_iter().map(|(_, n)| n).collect(),
        }
    } else {
        vparts.pop().unwrap().1
    }
}

//node structure is recursive grid
#[derive(Debug)]
pub enum Node {
    Final { indices: Vec<usize> },
    Grid { x: Vec<f32>, y: Vec<f32>, cells: Vec<Node>, tag: NodeTag },
    Table { table: Table<Vec<usize>> },
}

impl Node {
    fn tag(&self) -> NodeTag {
        match *self {
            Node::Grid { tag, .. } => tag,
            Node::Table { .. } => NodeTag::Complex,
            Node::Final { .. } => NodeTag::Singleton,
        }
    }
    ///collects indices by walking the tree of nodes, extending the the out vector
    fn indices(&self, out: &mut Vec<usize>) {
        match *self {
            Node::Final { ref indices } => out.extend_from_slice(&indices),
            Node::Grid { ref cells, .. } => {
                for n in cells {
                    n.indices(out);
                }
            }
            Node::Table { ref table } => {
                out.extend(
                    table.values()
                        .flat_map(|v| v.value.iter())
                        .cloned()
                );
            }
        }
    }
    //can produce an empty node
    fn singleton(nodes: &[(RectF, usize)]) -> Self {
        Node::Final { indices: nodes.iter().map(|t| t.1).collect() }
    }
}

#[derive(PartialOrd, Ord, Eq, PartialEq, Clone, Copy, Debug)]
pub enum NodeTag {
    Singleton,
    Line,
    Paragraph,
    Complex,
}


///translating nested nodes into flows.
/// works by first maping the node indices to the spans, classify the spans based on text info
pub fn items(flow: &mut Flow, spans: &[TextSpan], node: &Node, x_anchor: f32) {
    // dbg!(x_anchor);
    match *node {
        Node::Final { ref indices } => {
            if indices.len() > 0 {
                let node_spans = indices.iter().flat_map(|&i| spans.get(i));
                let _bbox = node_spans.clone().map(|s| s.rect).reduce(|a, b| a.union_rect(b)).unwrap();
                let class = classify(node_spans.clone());
                let mut text = String::new();
                let words = concat_text(&mut text, node_spans);

                let joined_words = words.iter().map(|word| word.text.as_str()).join(" ");
                // dbg!(joined_words);
                // println!("{}", joined_words.yellow());

                let t = match class {
                    Class::Header => RunType::Header,
                    _ => RunType::Paragraph,
                };

                flow.add_line(words, t);
            }
        }
        Node::Grid { ref x, y: _, ref cells, tag } => {
            match tag {
                NodeTag::Singleton |
                NodeTag::Line => {
                    let mut indices = vec![];
                    node.indices(&mut indices);
                    let line_spans = indices.iter().flat_map(|&i| spans.get(i));


                    // let span_text: Vec<&str> = line_spans.clone().map(|span| span.text.as_str()).collect();
                    // // dbg!(span_text);


                    let _bbox: RectF = line_spans.clone().map(|s| s.rect).reduce(|a, b| a.union_rect(b)).unwrap().into();

                    let mut text = String::new();
                    // dbg!(&line_spans);
                    let words = concat_text(&mut text, line_spans.clone());
                    let class = classify(line_spans.clone());

                    let t = match class {
                        Class::Header => RunType::Header,
                        _ => RunType::Paragraph,
                    };
                    flow.add_line(words, t);
                }
                NodeTag::Paragraph => {
                    assert_eq!(x.len(), 0);
                    let mut lines: Vec<(RectF, usize)> = vec![];
                    let mut indices = vec![];
                    for n in cells {
                        let start = indices.len();
                        n.indices(&mut indices);
                        //if cell yields no indices
                        if indices.len() > start {
                            let cell_spans = indices[start..].iter().flat_map(|&i| spans.get(i));
                            let bbox = cell_spans.map(|s| s.rect).reduce(|a, b| a.union_rect(b)).unwrap().into();
                            lines.push((bbox, indices.len()));
                        }
                    }

                    //the spans for the paragraph
                    let para_spans = indices.iter().flat_map(|&i| spans.get(i));

                    let span_text: Vec<&str> = para_spans.clone().map(|span| span.text.as_str()).collect();

                    let class = classify(para_spans.clone());
                    let bbox = lines.iter().map(|t| t.0).reduce(|a, b| a.union_rect(b)).unwrap();
                    let line_height = avg(para_spans.map(|s| s.rect.height())).unwrap();
                    // a point just slightly along from the minimum x
                    let left_margin = bbox.min_x() + 0.5 * line_height;


                    let mut left = 0;
                    let mut right = 0;

                    for (line_bbox, _) in lines.iter() {
                        if line_bbox.min_x() >= left_margin {
                            right += 1;
                        } else {
                            left += 1;
                        }
                    }

                    // typically paragraphs are indented to the right and longer than 2 lines.
                    // then there will be a higher left count than right count.
                    let indent = left > right;

                    let mut para_start = 0;
                    let mut line_start = 0;
                    let mut text = String::new();
                    let mut para_bbox = RectF::default();
                    let mut flow_lines = vec![];
                    for &(line_bbox, end) in lines.iter() {
                        if line_start != 0 {
                            // if a line is indented (or outdented), it marks a new paragraph
                            if (line_bbox.min_x() >= left_margin) == indent {
                                flow.runs.push(Run {
                                    lines: take(&mut flow_lines),
                                    kind: match class {
                                        Class::Header => RunType::Header,
                                        _ => RunType::Paragraph
                                    },
                                });
                                para_start = line_start;
                            } else {
                                text.push('\n');
                            }
                        }
                        if end > line_start {
                            let words = concat_text(&mut text, indices[line_start..end].iter().flat_map(|&i| spans.get(i)));

                            if words.len() > 0 {
                                // println!("{}", words.iter().map(|word| word.text.as_str()).join(" ").magenta());
                                flow_lines.push(Line { words });
                            }
                        }
                        if para_start == line_start {
                            para_bbox = line_bbox;
                        } else {
                            para_bbox = para_bbox.union_rect(line_bbox);
                        }
                        line_start = end;
                    }

                    let joined_words = flow_lines.iter().map(|line| line.words.iter().map(|word| word.text.as_str()).join(" "));
                    let words : Vec<Word> = flow_lines.clone().into_iter().flat_map(|line| line.words).collect();
                    // dbg!(words);
                    // println!("{}", joined_words.blue());

                    flow.runs.push(Run {
                        lines: flow_lines,
                        kind: match class {
                            Class::Header => RunType::Header,
                            _ => RunType::Paragraph
                        },
                    });
                }
                NodeTag::Complex => {
                    // println!("{}", "complex here".red());
                    // dbg!(&x);
                    let x_anchors = once(x_anchor).chain(x.iter().cloned()).cycle();
                    for (node, x) in cells.iter().zip(x_anchors) {
                        items(flow, spans, node, x);
                    }
                    // println!("{}", "complex ends".green());
                }
            }
        }
        Node::Table { ref table } => {
            if let Some(_bbox) = table.values()
                .flat_map(|v| v.value.iter().flat_map(|&i| spans.get(i).map(|s| s.rect)))
                .reduce(|a, b| a.union_rect(b)) {
                let table = table.flat_map(|indices| {
                    if indices.len() == 0 {
                        None
                    } else {
                        let line_spans = indices.iter().flat_map(|&i| spans.get(i));
                        let bbox: RectF = line_spans.clone().map(|s| s.rect).reduce(|a, b| a.union_rect(b)).unwrap().into();

                        let mut text = String::new();
                        concat_text(&mut text, line_spans.clone());
                        Some(CellContent {
                            text,
                            rect: bbox.into(),
                        })
                    }
                });
                flow.add_table(table);
            }
        }
    }
}


///splits boxes into cells based on the x and y gaps present in the structure of the page, currently
/// determines the structure of the document based on a threshold derived from the maximum gap present
/// on the page.
///
/// Cells are then checked for structural splits
fn split(boxes: &mut [(RectF, usize)], spans: &[TextSpan], lines: &Lines) -> Node {
    if boxes.len() > 0 {
        let range_end = boxes.iter().map(|v| v.1).max().unwrap_or_else(|| dbg!(boxes.len()));
        let range_start = boxes.iter().map(|v| v.1).min().unwrap();
        // let mut text = spans[range_start..range_end].iter().map(|t| t.text.clone()).collect::<Vec<String>>().join(" ");
        let mut text = spans[range_start..range_end].iter().map(|t| t.text.clone());
        // println!("{}", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".green());
        // dbg!(text.join(" "));
        let mut rects = spans[range_start..range_end].iter().map(|t| t.rect.clone());
        let text_and_rects: Vec<(String, RectF)> = text.zip(rects).collect();
        // dbg!(text_and_rects);
    }


    // let text = boxes.iter().map(|(_, i)| spans.get(*i).unwrap().text.clone()).collect::<Vec<String>>().join(" ");
    let num_boxes = boxes.len();
    if num_boxes < 2 {
        return Node::singleton(boxes);
    }
    sort_occurence(boxes);
    // sort_x(boxes);
    let mut max_x_gap = dist_x(boxes, &spans);
    // sort_y(boxes);
    let max_y_gap = dist_y_log(boxes, &spans);
    // dbg!(max_x_gap,max_y_gap);
    // if let Some((a, b, c)) = max_y_gap.as_ref() {
    //     dbg!(&spans[*c].text);
    // }


    let x_y_ratio = 4.0;


    // dbg!(max_x_gap, max_y_gap);
    let max_gap = match (max_x_gap, max_y_gap) {
        (Some((x, _, _)), Some((y, _, _))) => x.max(y * x_y_ratio),
        (Some((x, _, _)), None) => x,
        (None, Some((y, _, _))) => y * x_y_ratio,
        (None, None) => {
            sort_x(boxes);
            return Node::singleton(boxes);
        }
    };

    //y threshold is used to define the size of the gaps that are large enough relative to the max gap
    //basically the entire algorithm rests on this threashold being derived correctly from the
    // maximum gap found in the entire page even the numbers used to calculate these thresholds could
    // be improved
    // dbg!(max_gap);
    let x_threshold = (max_gap * 0.5).max(1.0);
    let y_threshold = (max_gap * 0.7 / x_y_ratio);//.max(0.01);
    let mut cells = vec![];

    //gaps that exceed a threshold that is a ratio of the largest gap, if the max x gap is
    // excessively large then there wont be any gaps picked up by the gaps function.
    sort_y(boxes);
    let y_gaps_with_gap_length: Vec<(f32, f32)> = gaps(y_threshold, boxes, |r| (r.min_y(), r.max_y()))
        .collect();
    // dbg!(y_threshold, &y_gaps_with_gap_length);

    let y_gaps: Vec<f32> = y_gaps_with_gap_length.into_iter().map(|(a, b)| b).collect();


    sort_x(boxes);
    let mut x_gaps_with_gap_length: Vec<(f32, f32)> = gaps(x_threshold, boxes, |r| (r.min_x(), r.max_x()))
        .collect();
    // dbg!(x_threshold, &x_gaps_with_gap_length);

    let x_gaps: Vec<f32> = x_gaps_with_gap_length.into_iter().map(|(a, b)| b).collect();


    //no gaps indicates that within the structure of the boxes there are no gaps that exceed the
    //threshold set, this means that this is called on any cell that can be viewed as a self contained unit maybe
    if x_gaps.len() == 0 && y_gaps.len() == 0 {
        dbg!("overlapping line");
        if boxes.len() > 0 {
            let range_end = boxes.iter().map(|v| v.1).max().unwrap_or_else(|| dbg!(boxes.len()));
            let range_start = boxes.iter().map(|v| v.1).min().unwrap();
            let mut text = spans[range_start..range_end].iter().map(|t| t.text.clone()).collect::<Vec<String>>().join(" ");
            dbg!(text);
        }
        return overlapping_lines(boxes);
    }


    if x_gaps.len() > 1 && y_gaps.len() > 1 {
        dbg!("splitting for a table");
        return split2(boxes, spans, lines);
    }

    sort_y(boxes);

    //use the gaps in the y axis to split the boxes up based on the points at which the boxes exceed
    //the threshold set by the x_y_ratio and the largest gap in the page
    //problem for our case were the x
    for row in split_by(boxes, &y_gaps, |r| r.min_y()) {
        if row.len() > 0 {
            let range_end = row.iter().map(|v| v.1).max().unwrap();
            let range_start = row.iter().map(|v| v.1).min().unwrap();
            let mut text = spans[range_start..range_end].iter().map(|t| t.text.clone()).collect::<Vec<String>>().join(" ");
            dbg!(text);
        }
        if x_gaps.len() > 0 {
            sort_x(row);
            //if there are gaps in the x direction across the overall structure split across the
            // gaps and the treat each for splits again
            if row.len() > 0 {
                let range_end = row.iter().map(|v| v.1).max().unwrap();
                let range_start = row.iter().map(|v| v.1).min().unwrap();
                let mut text = spans[range_start..range_end].iter().map(|t| t.text.clone()).collect::<Vec<String>>().join(" ");
                dbg!(text);
            }
            for cell in split_by(row, &x_gaps, |r| r.min_x()) {
                sort_y(cell);
                assert!(cell.len() < num_boxes);
                dbg!("splits again");
                // dbg!(&x_gaps);
                // dbg!(&cell);
                if cell.len() > 0 {
                    let range_end = cell.iter().map(|v| v.1).max().unwrap_or_else(|| dbg!(cell.len()));
                    let range_start = cell.iter().map(|v| v.1).min().unwrap();
                    let mut text = spans[range_start..=range_end].iter().map(|t| t.text.clone()).collect::<Vec<String>>();//.join(" ");
                    dbg!(text);
                }
                for boxe in cell.iter() {
                    dbg!(&spans[boxe.1].text);
                }
                cells.push(split(cell, spans, lines));
            }
        } else {
            let range_end = row.iter().map(|v| v.1).max().unwrap();
            let range_start = row.iter().map(|v| v.1).min().unwrap();
            let mut text = spans[range_start..range_end].iter().map(|t| t.text.clone()).collect::<Vec<String>>().join(" ");
            dbg!(text);
            cells.push(split(row, spans, lines));
        }
    }

    assert!(x_gaps.len() > 0 || y_gaps.len() > 0);
    let tag = if y_gaps.len() == 0 {
        if cells.iter().all(|n| n.tag() <= NodeTag::Line) {
            NodeTag::Line
        } else {
            NodeTag::Complex
        }
    } else if x_gaps.len() == 0 {
        if cells.iter().all(|n| n.tag() <= NodeTag::Line) {
            NodeTag::Paragraph
        } else {
            NodeTag::Complex
        }
    } else {
        NodeTag::Complex
    };

    Node::Grid {
        x: x_gaps,
        y: y_gaps,
        cells,
        tag,
    }
}

#[allow(dead_code)]
fn split_v(boxes: &mut [(RectF, usize)]) -> Node {
    let num_boxes = boxes.len();
    if num_boxes < 2 {
        return Node::singleton(boxes);
    }

    let mut max_y_gap = dist_y(boxes);


    let max_gap = match max_y_gap {
        Some((y, _, _)) => y,
        None => {
            sort_x(boxes);
            return Node::singleton(boxes);
        }
    };
    let threshold = max_gap * 0.8;
    let mut cells = vec![];

    let y_gaps_with_gap_length: (Vec<(f32, f32)>) = gaps(threshold, boxes, |r| (r.min_y(), r.max_y()))
        .collect();

    let y_gaps: Vec<f32> = y_gaps_with_gap_length.into_iter().map(|(a, b)| b).collect();

    for row in split_by(boxes, &y_gaps, |r| r.min_y()) {
        assert!(row.len() < num_boxes);
        cells.push(split_v(row));
    }

    let tag = if cells.iter().all(|n| n.tag() <= NodeTag::Line) {
        NodeTag::Paragraph
    } else {
        NodeTag::Complex
    };

    Node::Grid {
        x: vec![],
        y: y_gaps,
        cells,
        tag,
    }
}

///
fn top_bottom_gap(boxes: &mut [(RectF, usize)], bbox: RectF) -> (Option<usize>, Option<usize>) {
    let num_boxes = boxes.len();
    if num_boxes < 2 {
        return (None, None);
    }
    println!("{}", "listing top bottom gaps".red());
    let mut gaps = gap_list(boxes, |r| (r.min_y(), r.max_y()));
    //TODO ask why these numbers?
    let top_limit = bbox.min_y() + bbox.height() * 0.3;
    let bottom_limit = bbox.min_y() + bbox.height() * 0.8;
    // looks at the fist gap to see whether the start of the first rect is  above the top limit
    // returns the index in a tuple if either is above or below the respective limits
    //only returns info on footer if there is a header
    match gaps.next() {
        Some((y, _, top)) if y < top_limit => {
            match gaps.last() {
                Some((y, _, bottom)) if y > bottom_limit => (Some(top), Some(bottom)),
                _ => (Some(top), None)
            }
        }
        Some((y, _, bottom)) if y > bottom_limit => (None, Some(bottom)),
        // _ => (None, None)
        _ => {
            match gaps.last() {
                Some((y, _, bottom)) if y > bottom_limit => (None, Some(bottom)),
                _ => (None, None)
            }
        }
    }
}

fn left_right_gap(boxes: &mut [(RectF, usize)], bbox: RectF) -> (Option<usize>, Option<usize>) {
    let num_boxes = boxes.len();
    if num_boxes < 2 {
        return (None, None);
    }
    println!("{}", "listing  left right gaps".red());

    let mut gaps = gap_list(boxes, |r| (r.min_x(), r.max_x()));
    let left_limit = bbox.min_x() + bbox.width() * 0.1;
    let right_limit = bbox.min_x() + bbox.width() * 0.8;
    match gaps.next() {
        Some((x, _, left)) if x < left_limit => {
            match gaps.last() {
                Some((x, _, right)) if x > right_limit => (Some(left), Some(right)),
                _ => (Some(left), None)
            }
        }
        Some((x, _, right)) if x > right_limit => (None, Some(right)),
        _ => (None, None)
    }
}

///sort by left_most x value
fn sort_x(boxes: &mut [(RectF, usize)]) {
    boxes.sort_unstable_by(|a, b| a.0.min_x().partial_cmp(&b.0.min_x()).unwrap());
}

///sort by top most y value
fn sort_y(boxes: &mut [(RectF, usize)]) {
    boxes.sort_unstable_by(|a, b| a.0.min_y().partial_cmp(&b.0.min_y()).unwrap());
}

fn sort_occurence(boxes: &mut [(RectF, usize)]) {
    boxes.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
}


fn overlapping_lines(boxes: &mut [(RectF, usize)]) -> Node {
    sort_y(boxes);
    let avg_height = avg(boxes.iter().map(|(r, _)| r.height())).unwrap();

    let mut y_center = boxes[0].0.center().y();
    let mut lines = vec![];
    let mut y_splits = vec![];

    let mut start = 0;
    'a: loop {
        for (i, &(r, _)) in boxes[start..].iter().enumerate() {
            if r.center().y() > 0.5 * avg_height + y_center {
                let end = start + i;
                sort_x(&mut boxes[start..end]);
                let bbox = boxes[start..end].iter().map(|&(r, _)| r).reduce(|a, b| a.union_rect(b)).unwrap();

                y_splits.push(bbox.max_y());
                lines.push(Node::singleton(&boxes[start..end]));
                y_center = r.center().y();

                start = end;
                continue 'a;
            }
        }

        sort_x(&mut boxes[start..]);
        lines.push(Node::singleton(&boxes[start..]));

        break;
    }
    match lines.len() {
        0 => Node::singleton(&[]),
        1 => lines.pop().unwrap(),
        _ => Node::Grid {
            x: vec![],
            y: y_splits,
            cells: lines,
            tag: NodeTag::Paragraph,
        }
    }
}


///returns an iterator over a tuple of gaps between a section of boxes represented by the start and
///the end of the gap, indexed from 1
///produces None values for boxes with minimums that are the same as or less than the previous one.
fn gap_list_log<'a>(boxes: &'a [(RectF, usize)], span: impl Fn(&RectF) -> (f32, f32) + 'a, spans: &'a [TextSpan]) -> impl Iterator<Item=(f32, f32, usize)> + 'a {
    println!("{}", "listing the gaps".red());
    // dbg!(&boxes);
    let mut boxes = boxes.iter();
    let &(ref r, i) = boxes.next().unwrap();
    let (last_min, mut last_max) = span(r);
    // dbg!(last_min,last_max, &spans[i].text);
    boxes.enumerate().filter_map(move |(idx, &(ref r, _))| {
        let (min, max) = span(&r);
        // dbg!(&min, &max, &last_max, &spans[idx].text, &spans[idx + 1].text);
        let r = if min > last_max {
            // dbg!("some");
            Some((last_max, min, idx + 1))
        } else {
            // dbg!("None");
            None
        };
        last_max = max.max(last_max);
        r
    })
}


fn gap_list<'a>(boxes: &'a [(RectF, usize)], span: impl Fn(&RectF) -> (f32, f32) + 'a) -> impl Iterator<Item=(f32, f32, usize)> + 'a {
    println!("{}", "listing the gaps".red());
    let mut boxes = boxes.iter();
    let &(ref r, _) = boxes.next().unwrap();
    let (_, mut last_max) = span(r);
    boxes.enumerate().filter_map(move |(idx, &(ref r, _))| {
        let (min, max) = span(&r);
        // dbg!(&min, &max);
        let r = if min > last_max {
            // dbg!("some");
            Some((last_max, min, idx + 1))
        } else {
            // dbg!("None");
            None
        };
        last_max = max.max(last_max);
        r
    })
}

///Returns an iterator of
///function depends on a span function that determines the direction of the gaps by returning a
/// tuple of either x or y values that describe the height or the width of the box
/// to test whether there exists any gaps that exceed a threshold mapping the center point to a new iterator
fn gaps<'a>(threshold: f32, boxes: &'a [(RectF, usize)], span: impl Fn(&RectF) -> (f32, f32) + 'a) -> impl Iterator<Item=(f32, f32)> + 'a {
    let mut boxes = boxes.iter();

    let &(ref r, _) = boxes.next().unwrap();
    let (_, mut last_max) = span(r);
    boxes.filter_map(move |&(ref r, _)| {
        let (min, max) = span(&r);
        let r = if min - last_max >= threshold {
            Some((min - last_max, 0.5 * (last_max + min)))
        } else {
            None
        };
        last_max = max.max(last_max);
        r
    })
}

/// takes a slice of boxes and a span function that determines the axis of interest and
/// returns the magnitude of the largest gap and the centre point along the same axis.
fn max_gap(boxes: &[(RectF, usize)], span: impl Fn(&RectF) -> (f32, f32)) -> Option<(f32, f32, usize)> {
    let gap_list = gap_list(boxes, span);


    gap_list.map(|(a, b, c)| {
        let q = (&b - &a, 0.5 * (&a + &b));

        // dbg!(boxes[c -1].0.0, boxes[c].0.0);
        // dbg!(q,c);

        (a, b, c)
    }).max_by_key(|&(a, b, c)| NotNan::new(b - a).unwrap())//finds the largest gap by subtracting last from next value as an iterator
        .map(|(a, b, c)| (b - a, 0.5 * (a + b), c))// maps an option to the gap distance and the centre point of the gap
}

fn max_gap_log(boxes: &[(RectF, usize)], span: impl Fn(&RectF) -> (f32, f32), spans: &[TextSpan]) -> Option<(f32, f32, usize)> {
    let gap_list = gap_list_log(boxes, span, spans);


    gap_list.map(|(a, b, c)| {
        let q = (&b - &a, 0.5 * (&a + &b));

        // dbg!(boxes[c -1].0.0, boxes[c].0.0);
        // dbg!(q,c);

        (a, b, c)
    }).max_by_key(|&(a, b, c)| NotNan::new(b - a).unwrap())//finds the largest gap by subtracting last from next value as an iterator
        .map(|(a, b, c)| (b - a, 0.5 * (a + b), c))// maps an option to the gap distance and the centre point of the gap
}

///returns the gap distances and the x value for the centre point of the largest gap in the x axis
/// will return None values for boxes that are contiguously placed in space, i.e: line phrases or paragraphs
/// or if the spacing is equal
fn dist_x(boxes: &[(RectF, usize)], spans: &[TextSpan]) -> Option<(f32, f32, usize)> {
    println!("{}", "max x".red());
    max_gap_log(boxes, |r| (r.min_x(), r.max_x()), spans)
}

///returns the gap distances and the xyvalue for the centre point of the largest gap in the y axis
/// will return None values for boxes that are contiguously placed in space, i.e: line phrases or paragraphs
/// or if the spacing is equal
///
fn dist_y_log(boxes: &[(RectF, usize)], spans: &[TextSpan]) -> Option<(f32, f32, usize)> {
    println!("{}", "max y".red());
    max_gap_log(boxes, |r| (r.min_y(), r.max_y()), spans)
}

fn dist_y(boxes: &[(RectF, usize)]) -> Option<(f32, f32, usize)> {
    println!("{}", "max y".red());
    max_gap(boxes, |r| (r.min_y(), r.max_y()))
}

///by is a function that informs the direction in which spans are created
/// data is a list of rects indexed by their appearance in the page
/// points are point in the page in which gaps exist
fn split_by<'a>(list: &'a mut [(RectF, usize)], at: &'a [f32], by: impl Fn(&RectF) -> f32) -> impl Iterator<Item=&'a mut [(RectF, usize)]> {
    SplitBy {
        data: list,
        points: at.iter().cloned(),
        by,
        end: false,
    }
}

///an iterator that operates on a mutable slice of boxes, calls to next() applies by function to
/// find a position at which to split the slice into a head and tail, the head value is returned
/// from the call to next(), the tail is set as the data for subsequent calls to next to operate on
struct SplitBy<'a, I, F> {
    data: &'a mut [(RectF, usize)],
    points: I,
    by: F,
    end: bool,
}

impl<'a, I, F> Iterator for SplitBy<'a, I, F> where
    I: Iterator<Item=f32>,
    F: Fn(&RectF) -> f32
{
    type Item = &'a mut [(RectF, usize)];
    fn next(&mut self) -> Option<Self::Item> {
        if self.end {
            return None;
        }
        match self.points.next() {
            Some(p) => {
                //gets the position of the first point that is past the gap point
                let idx = self.data.iter().position(|(ref r, _)| (self.by)(r) > p).unwrap_or(self.data.len());
                // idx is used to split data into a head and a tail, tail is returned to self and head is
                let (head, tail) = take(&mut self.data).split_at_mut(idx);
                self.data = tail;
                Some(head)
            }
            // if no points left then return last tail
            None => {
                self.end = true;
                Some(take(&mut self.data))
            }
        }
    }
}

use super::util::Tri;

#[derive(Copy, Clone, Debug, PartialEq)]
enum Class {
    Number,
    Header,
    Paragraph,
    Mixed,
}

///used for holding information about spans
#[derive(Debug)]
struct TriCount {
    tru: usize,
    fal: usize,
}

impl TriCount {
    fn new() -> Self {
        TriCount {
            tru: 0,
            fal: 0,
        }
    }
    ///adds count for true or false based on a check on the text span properties
    fn add(&mut self, b: bool) {
        match b {
            false => self.fal += 1,
            true => self.tru += 1,
        }
    }
    ///Defines the state of the
    fn count(&self) -> Tri {
        match (self.fal, self.tru) {
            (0, 0) => Tri::Unknown,
            (0, _) => Tri::True,
            (_, 0) => Tri::False,
            (f, t) => Tri::Maybe(t as f32 / (t + f) as f32)
        }
    }
}


//will classigy a set of spans based on whether or not the encountered text has font size or
fn classify<'a>(spans: impl Iterator<Item=&'a TextSpan>) -> Class {
    use pdf_render::FontEntry;

    let mut bold = TriCount::new();
    let mut numeric = TriCount::new();
    let mut uniform = TriCount::new();
    let mut first_font: *const FontEntry = std::ptr::null();


    //Spans checked for
    for s in spans {
        //Checks if span is numeric
        numeric.add(is_number(&s.text));
        if let Some(ref font) = s.font {
            //checks for font name bold
            bold.add(font.name.contains("Bold"));
            //
            let font_ptr = Arc::as_ptr(font);
            if first_font.is_null() {
                first_font = font_ptr;
            } else {
                uniform.add(font_ptr == first_font);
            }
        }
    }

    //why this
    uniform.add(true);

    match (numeric.count(), bold.count(), uniform.count()) {
        (Tri::True, _, Tri::True) => Class::Number,
        (_, Tri::True, Tri::True) => Class::Header,
        (_, Tri::False, Tri::True) => Class::Paragraph,
        (_, Tri::False, _) => Class::Paragraph,
        (_, Tri::Maybe(_), _) => Class::Paragraph,
        _ => Class::Mixed
    }
}
