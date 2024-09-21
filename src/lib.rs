use multimap::MultiMap;
use rand::seq::SliceRandom;
use std::collections::HashSet;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

struct MyImage {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    channels: u32,
}

impl MyImage {
    pub fn new(width: u32, height: u32, channels: u32) -> MyImage {
        let mut image = MyImage {
            pixels: vec![0; (width * height * 4) as usize],
            width,
            height,
            channels,
        };
        for idx in (channels - 1..width * height * channels).step_by(channels as usize) {
            image.pixels[idx as usize] = 255;
        }
        return image;
    }

    pub fn set_white(&mut self, u: u32, v: u32) {
        self.pixels[(self.width * self.channels * v + self.channels * u + 0) as usize] = 255;
        self.pixels[(self.width * self.channels * v + self.channels * u + 1) as usize] = 255;
        self.pixels[(self.width * self.channels * v + self.channels * u + 2) as usize] = 255;
    }
}

struct Maze {
    image: MyImage,
    width: u32,
    height: u32,
    cell_size: u32,
}

impl Maze {
    pub fn new(width: u32, height: u32, cell_size: u32) -> Maze {
        let image = MyImage::new(width * (cell_size + 1) + 1, height * (cell_size + 1) + 1, 4);

        let mut maze = Maze {
            image,
            width,
            height,
            cell_size,
        };
        for x in 0..width {
            for y in 0..height {
                maze.set_cell_white(x, y);
            }
        }
        return maze;
    }

    fn set_cell_white(&mut self, x: u32, y: u32) {
        for u in 0..self.cell_size {
            for v in 0..self.cell_size {
                self.image.set_white(
                    u + x * (self.cell_size + 1) + 1,
                    v + y * (self.cell_size + 1) + 1,
                );
            }
        }
    }

    fn set_openings_white(&mut self) {
        for u in 0..self.cell_size {
            self.image.set_white(u + 1, 0);
            self.image.set_white(
                (self.width - 1) * (self.cell_size + 1) + u + 1,
                self.height * (self.cell_size + 1),
            );
        }
    }

    fn set_wall_white(&mut self, first: &(u32, u32), second: &(u32, u32)) {
        for u in 0..self.cell_size {
            if second.0 > first.0 {
                self.image.set_white(
                    self.cell_size + first.0 * (self.cell_size + 1) + 1,
                    u + first.1 * (self.cell_size + 1) + 1,
                );
            } else if first.0 > second.0 {
                self.image.set_white(
                    self.cell_size + second.0 * (self.cell_size + 1) + 1,
                    u + second.1 * (self.cell_size + 1) + 1,
                );
            } else if second.1 > first.1 {
                self.image.set_white(
                    u + first.0 * (self.cell_size + 1) + 1,
                    self.cell_size + first.1 * (self.cell_size + 1) + 1,
                );
            } else if first.1 > second.1 {
                self.image.set_white(
                    u + second.0 * (self.cell_size + 1) + 1,
                    self.cell_size + second.1 * (self.cell_size + 1) + 1,
                );
            }
        }
    }

    pub fn get_image(&self) -> &MyImage {
        return &self.image;
    }
}

#[wasm_bindgen]
pub fn generate(width: u32, height: u32, cell_size: u32) {
    let mut maze = Maze::new(width, height, cell_size);

    let mut graph = MultiMap::new();
    for x in 0..width {
        for y in 0..height {
            // add vertical edges
            if y < height - 1 {
                graph.insert((y, x), (y + 1, x));
                graph.insert((y + 1, x), (y, x));
            }
            // add horizontal edges
            if x < width - 1 {
                graph.insert((y, x), (y, x + 1));
                graph.insert((y, x + 1), (y, x));
            }
        }
    }

    let mut rng = rand::thread_rng();

    let mut unchosen_vertices = HashSet::new();
    for vertex in graph.keys() {
        unchosen_vertices.insert(vertex);
    }

    let mut chosen_vertices = HashSet::new();

    let x = unchosen_vertices.clone().into_iter().collect::<Vec<_>>();
    let chosen = x.choose(&mut rng).expect("there should really be one...");
    chosen_vertices.insert(*chosen);
    unchosen_vertices.remove(chosen);

    loop {
        // random walk (with loop erasure) from new point until we hit something in chosen_vertices
        let mut path = Vec::new();
        let x = unchosen_vertices.clone().into_iter().collect::<Vec<_>>();
        let start = x.choose(&mut rng).expect("there should really be one...");

        path.push(*start);
        loop {
            let start = path.last().expect("should not be empty...");
            let candidates = graph.get_vec(start).expect("key should really exist...");
            let next = candidates
                .choose(&mut rng)
                .expect("there should really be one...");
            path.push(next);
            if chosen_vertices.contains(&next) {
                break;
            }
        }
        path = erase_loops(path);

        for idx in 0..path.len() - 1 {
            maze.set_wall_white(path[idx], path[idx + 1]);
        }
        for vertex in &path {
            unchosen_vertices.remove(vertex);
        }
        chosen_vertices.extend(path.iter().cloned().collect::<HashSet<_>>());
        if unchosen_vertices.is_empty() {
            break;
        }
    }

    maze.set_openings_white();

    let array = &maze.get_image().pixels.clone();
    let image_data_temp: ImageData = ImageData::new_with_u8_clamped_array_and_sh(
        Clamped(&array),
        maze.get_image().width,
        maze.get_image().height,
    )
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
    context.put_image_data(&image_data_temp, 0.0, 0.0).unwrap();
}

fn erase_loops(path: Vec<&(u32, u32)>) -> Vec<&(u32, u32)> {
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
