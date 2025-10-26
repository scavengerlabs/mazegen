use rand::prelude::IndexedRandom;
use rand::Rng;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

mod fortunes;

struct MyImage {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    channels: u32,
}

struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl MyImage {
    pub fn new(width: u32, height: u32, channels: u32, color: &Color) -> MyImage {
        let mut image = MyImage {
            pixels: vec![255; (width * height * 4) as usize],
            width,
            height,
            channels,
        };

        for u in 0..image.width {
            for v in 0..image.height {
                image.set_px(u, v, color);
            }
        }
        return image;
    }

    pub fn set_px(&mut self, u: u32, v: u32, color: &Color) {
        if u >= self.width || v >= self.height {
            return;
        }
        self.pixels[(self.width * self.channels * v + self.channels * u + 0) as usize] = color.r;
        self.pixels[(self.width * self.channels * v + self.channels * u + 1) as usize] = color.g;
        self.pixels[(self.width * self.channels * v + self.channels * u + 2) as usize] = color.b;
    }

    pub fn plot_line_segment(&mut self, first: &Point, second: &Point, color: &Color) {
        let x1 = first.x;
        let x2 = second.x;
        let y1 = first.y;
        let y2 = second.y;
        let dx = x2 - x1;
        let dy = y2 - y1;
        if dx.abs() > dy.abs() {
            let u1 = x1.round() as u32;
            let u2 = x2.round() as u32;
            let u_min = std::cmp::min(u1, u2);
            let u_max = std::cmp::max(u1, u2);
            for u in u_min..u_max {
                let v = ((y2 - y1) / (x2 - x1) * (u as f32 - x1) + y1).round() as u32;
                self.set_px(u, v, color);
            }
        } else {
            let v1 = y1.round() as u32;
            let v2 = y2.round() as u32;
            let v_min = std::cmp::min(v1, v2);
            let v_max = std::cmp::max(v1, v2);
            for v in v_min..v_max {
                let u = ((x2 - x1) / (y2 - y1) * (v as f32 - y1) + x1).round() as u32;
                self.set_px(u, v, color);
            }
        }
    }
}

struct Maze {
    image: MyImage,
    width: u32,
    height: u32,
    cell_size: u32,
}

struct Point {
    id: Uuid,
    x: f32,
    y: f32,
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Point {}

impl Point {
    pub fn new(x: f32, y: f32) -> Point {
        return Point {
            id: Uuid::new_v4(),
            x: x,
            y: y,
        };
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct Pixel {
    u: u32,
    v: u32,
}

impl Point {
    pub fn nearest_pixel(&self) -> Pixel {
        return Pixel {
            u: self.x.round() as u32,
            v: self.y.round() as u32,
        };
    }
}

impl Maze {
    pub fn new(width: u32, height: u32, cell_size: u32) -> Maze {
        let mut image = MyImage::new(
            width * (cell_size + 1) + 1,
            height * (cell_size + 1) + 1,
            4,
            &Color { r: 0, g: 0, b: 0 },
        );

        let points = vec![
            Point::new(0.0, 0.0),
            Point::new((width * (cell_size + 1)) as f32, 0.0),
            Point::new(
                (width * (cell_size + 1)) as f32,
                (height * (cell_size + 1)) as f32,
            ),
            Point::new(0.0, (height * (cell_size + 1)) as f32),
            Point::new(0.0, 0.0),
        ];
        let polyline = PolyLine { points: points };
        polyline.plot(&mut image, &Color { r: 255, g: 0, b: 0 });

        let maze = Maze {
            image,
            width,
            height,
            cell_size,
        };
        return maze;
    }

    fn plot_divider(&mut self, first: &Point, second: &Point) {
        let mu = (first.x + second.x + 1.0) / 2.0 * (self.cell_size + 1) as f32;
        let mv = (first.y + second.y + 1.0) / 2.0 * (self.cell_size + 1) as f32;
        let dv = (first.x - second.x) * (self.cell_size + 1) as f32;
        let du = (first.y - second.y) * (self.cell_size + 1) as f32;
        let pt0 = Point::new(mu + (du / 2.0 - 0.5), mv + (dv / 2.0 - 0.5));
        let pt1 = Point::new(mu - (du / 2.0 - 0.5), mv - (dv / 2.0 - 0.5));
        self.image.plot_line_segment(
            &pt0,
            &pt1,
            &Color {
                r: 255,
                g: 0,
                b: 255,
            },
        );
    }

    pub fn get_image(&self) -> &MyImage {
        return &self.image;
    }
}

#[wasm_bindgen]
pub fn generate(width: u32, height: u32, cell_size: u32) {
    let mut maze = Maze::new(width, height, cell_size);

    let mut rng = rand::rng();

    let mut graph = HashMap::new();
    let mut centers = HashMap::new();
    for u in 0..width {
        for v in 0..height {
            let pt = Point::new(
                u as f32 + rng.random::<f32>() / 5.0,
                v as f32 + rng.random::<f32>() / 5.0,
            );
            centers.insert(pt.id, pt);
        }
    }
    let mut center_iter = centers.values();
    while let Some(px0) = center_iter.next() {
        let mut center_iter2 = center_iter.clone();
        while let Some(px1) = center_iter2.next() {
            if ((px0.x as i32) - (px1.x as i32)).abs() == 1 && (px0.y as i32) - (px1.y as i32) == 0
            {
                graph
                    .entry(px0.id)
                    .or_insert(HashSet::new())
                    .insert(&px1.id);
                graph
                    .entry(px1.id)
                    .or_insert(HashSet::new())
                    .insert(&px0.id);
            } else if ((px0.y as i32) - (px1.y as i32)).abs() == 1
                && (px0.x as i32) - (px1.x as i32) == 0
            {
                graph
                    .entry(px0.id)
                    .or_insert(HashSet::new())
                    .insert(&px1.id);
                graph
                    .entry(px1.id)
                    .or_insert(HashSet::new())
                    .insert(&px0.id);
            }
        }
    }

    let mut unchosen_vertices = HashSet::new();
    for vertex in graph.keys() {
        unchosen_vertices.insert(vertex);
    }

    let mut chosen_vertices = HashSet::new();

    let x = unchosen_vertices.clone().into_iter().collect::<Vec<_>>();
    let chosen = x.choose(&mut rng).expect("there should really be one...");
    chosen_vertices.insert(*chosen);
    unchosen_vertices.remove(chosen);

    let mut connections = HashSet::new();
    loop {
        // random walk (with loop erasure) from new point until we hit something in chosen_vertices
        let mut path = Vec::new();
        let x = unchosen_vertices.clone().into_iter().collect::<Vec<_>>();
        let start = x.choose(&mut rng).expect("there should really be one...");

        path.push(*start);
        loop {
            let start = path.last().expect("should not be empty...");
            let candidates = graph
                .get(start)
                .expect("key should really exist...")
                .clone()
                .into_iter()
                .collect::<Vec<_>>();
            let next = candidates
                .choose(&mut rng)
                .expect("there should really be one...");
            path.push(next);
            if chosen_vertices.contains(next) {
                break;
            }
        }
        path = erase_loops(path);

        for idx in 0..path.len() - 1 {
            connections.insert((path[idx], path[idx + 1]));
        }
        for vertex in &path {
            unchosen_vertices.remove(vertex);
        }
        chosen_vertices.extend(path.iter().cloned().collect::<HashSet<_>>());
        if unchosen_vertices.is_empty() {
            break;
        }
    }
    for (source, destinations) in graph.iter() {
        for destination in destinations {
            if connections.contains(&(source, destination))
                || connections.contains(&(destination, source))
            {
                continue;
            }
            maze.plot_divider(
                centers.get(source).expect("this should really be here..."),
                centers
                    .get(destination)
                    .expect("this should really be here..."),
            );
        }
    }

    let image = maze.get_image();
    render_to_canvas(image);
}

fn render_to_canvas(image: &MyImage) {
    let array = image.pixels.clone();
    let image_data_temp: ImageData =
        ImageData::new_with_u8_clamped_array_and_sh(Clamped(&array), image.width, image.height)
            .unwrap();

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    let context: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();
    context.reset();
    context.put_image_data(&image_data_temp, 0.0, 0.0).unwrap();
}

fn erase_loops<T: std::cmp::PartialEq>(path: Vec<&T>) -> Vec<&T> {
    let mut indices = Vec::new();
    let mut idx = 0;
    indices.push(idx);
    while idx < path.len() - 1 {
        for (pos, vertex) in path[idx..path.len() - 1].iter().enumerate().rev() {
            if *vertex == path[idx] {
                idx += pos + 1;
                indices.push(idx);
                break;
            }
        }
    }
    let mut loop_erased_path = Vec::new();
    for idx in indices {
        loop_erased_path.push(path[idx]);
    }
    return loop_erased_path;
}

struct PolyLine {
    points: Vec<Point>,
}

impl PolyLine {
    pub fn plot(&self, image: &mut MyImage, color: &Color) {
        let mut origin = &self.points[0];
        for destination in &self.points[1..self.points.len()] {
            image.plot_line_segment(origin, destination, color);
            origin = destination;
        }
    }
}
