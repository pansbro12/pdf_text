use pathfinder_geometry::rect::RectF;
use std::collections::HashSet;

use entry::Flow;
use pdf::{
    backend::Backend,
    object::{Page, Resolve},
    PdfError,
};
use pdf_render::{
    render_page, render_pattern,
    tracer::{DrawItem, TraceCache, Tracer},
    Fill, FillMode,
};

pub mod entry;
mod text;
mod tree;
<<<<<<< HEAD
pub mod util;

pub fn run<B: Backend>(
    file: &pdf::file::CachedFile<B>,
    page: &Page,
    resolve: &impl Resolve,
) -> Result<Flow, PdfError> {
=======
mod util;
mod text;
pub mod entry;

pub fn run<B: Backend>(file: &pdf::file::CachedFile<B>, page: &Page, resolve: &impl Resolve) -> Result<Flow, PdfError> {
>>>>>>> b61e68d (update pdf)
    let cache = TraceCache::new();

    let mut clip_paths = vec![];
    let mut tracer = Tracer::new(&cache, &mut clip_paths);

    render_page(&mut tracer, resolve, &page, Default::default())?;

    let bbox: RectF = tracer.view_box();

    let items = tracer.finish();
    let mut patterns = HashSet::new();
    for item in items.iter() {
        if let DrawItem::Vector(ref v) = item {
            if let Some(FillMode {
                color: Fill::Pattern(id),
                ..
            }) = v.fill
            {
                patterns.insert(id);
            }
            if let Some((
                FillMode {
                    color: Fill::Pattern(id),
                    ..
                },
                _,
            )) = v.stroke
            {
                patterns.insert(id);
            }
        }
    }

    let mut spans = vec![];
    let mut lines = vec![];
<<<<<<< HEAD

    let mut visit_item = |item| match item {
        DrawItem::Text(t, _) if bbox.intersects(t.rect) => {
            spans.push(t);
        }
        DrawItem::Vector(path) if bbox.intersects(path.outline.bounds()) => {
            for contour in path.outline.contours() {
                use pathfinder_content::{outline::ContourIterFlags, segment::SegmentKind};
                for segment in contour.iter(ContourIterFlags::empty()) {
                    match segment.kind {
                        SegmentKind::Line => lines.push([
                            segment.baseline.from_x(),
                            segment.baseline.from_y(),
                            segment.baseline.to_x(),
                            segment.baseline.to_y(),
                        ]),
                        _ => {}
=======
    let mut visit_item = |item| {
        match item {
            DrawItem::Text(t) if bbox.intersects(t.rect) => {

                if t.text.chars().any(|c| !c.is_whitespace()) {
                    spans.push(t);
                }
            }
            DrawItem::Vector(path) if bbox.intersects(path.outline.bounds()) => {
                for contour in path.outline.contours() {
                    use pathfinder_content::{outline::ContourIterFlags, segment::SegmentKind};
                    for segment in contour.iter(ContourIterFlags::empty()) {
                        match segment.kind {
                            SegmentKind::Line => lines.push([
                                segment.baseline.from_x(),
                                segment.baseline.from_y(),
                                segment.baseline.to_x(),
                                segment.baseline.to_y()
                            ]),
                            _ => {}
                        }
>>>>>>> b61e68d (update pdf)
                    }
                }
            }
        }
        _ => {}
    };

    for &p in patterns.iter() {
        let pattern = match resolve.get(p) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("failed to load pattern: {:?}", e);
                continue;
            }
        };
<<<<<<< HEAD
        let mut pat_tracer = Tracer::new(&cache, &mut clip_paths);

        render_pattern(&mut pat_tracer, &*pattern, resolve)?;

=======
        let mut pat_tracer = Tracer::new(&cache);

        render_pattern(&mut pat_tracer, &*pattern, resolve)?;
>>>>>>> b61e68d (update pdf)
        let pat_items = pat_tracer.finish();
        for item in pat_items {
            visit_item(item);
        }
    }

    for item in items {
        visit_item(item);
    }
<<<<<<< HEAD
    //creates a root node for the page
    let root = tree::build(&spans, bbox, &lines);
=======
>>>>>>> b61e68d (update pdf)

    let root = tree::build(&spans, bbox, &lines);
    let mut flow = Flow::new();
    tree::items(&mut flow, &spans, &root, bbox.min_x());
    Ok(flow)
}