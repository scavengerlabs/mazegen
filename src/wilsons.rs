use rand::prelude::IndexedRandom;
use std::collections::HashMap;
use std::collections::HashSet;
use uuid::Uuid;

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

pub fn wilsons<T: std::cmp::Eq + std::hash::Hash + Copy>(
    graph: &HashMap<T, HashSet<&T>>,
) -> HashSet<(T, T)> {
    let mut rng = rand::rng();

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
            connections.insert((*path[idx], *path[idx + 1]));
        }
        for vertex in &path {
            unchosen_vertices.remove(vertex);
        }
        chosen_vertices.extend(path.iter().cloned().collect::<HashSet<_>>());
        if unchosen_vertices.is_empty() {
            break;
        }
    }
    return connections;
}
