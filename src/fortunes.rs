use rand::Rng;
use std::cmp;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fmt;
use std::ops::Neg;
use std::sync::atomic::{AtomicU32, Ordering};
use svg::node::element::path::Data;
use svg::node::element::Circle;
use svg::node::element::Path;
use svg::Document;

// binary tree with ids for arcs and edges, per Fortune's algorithm
// map from arc id to adjacent edges
// map from edge id to adjacent arcs

#[derive(Copy, Clone, Debug)]
pub struct Point {
    x: f32,
    y: f32,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        return Point { x: x, y: y };
    }

    pub fn distance_from(&self, other: &Point) -> f32 {
        // compute Euclidean distance between points
        return ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).sqrt();
    }

    pub fn perpendicular(&self, other: &Point) -> Direction {
        // get the direction perpendicular to the line connecting two points,
        // pointed right
        // TODO: this may not be necessary
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        if dy >= 0.0 {
            return Direction::new(dy, -dx);
        } else {
            return Direction::new(-dy, dx);
        }
    }
}

#[derive(Debug)]
struct LineSegment {
    first: Point,
    second: Point,
}

impl LineSegment {
    pub fn length(&self) -> f32 {
        return ((self.first.x - self.second.x).powi(2) + (self.first.y - self.second.y).powi(2))
            .sqrt();
    }
    pub fn intersection(&self, ray: Ray) -> Option<Point> {
        let segment_ray = Ray {
            start: self.first,
            direction: Direction::new(self.second.x - self.first.x, self.second.y - self.first.y),
        };
        match ray.intersection(segment_ray) {
            Some(ray_intersection) => {
                let segment_ray_length = segment_ray.project(&ray_intersection);
                if segment_ray_length > self.length() {
                    return None;
                }
                return Some(ray_intersection);
            }
            None => {
                return None;
            }
        }
    }
}

struct Polyline {
    points: Vec<Point>,
}

impl Polyline {
    pub fn new() -> Self {
        return Polyline { points: vec![] };
    }

    pub fn nearest_intersection(&self, ray: Ray) -> Option<Point> {
        let mut output = None;
        let mut min_distance = f32::MAX;
        for idx in 0..self.points.len() - 1 {
            let line_segment = LineSegment {
                first: self.points[idx],
                second: self.points[idx + 1],
            };
            match line_segment.intersection(ray) {
                Some(point) => {
                    let distance = ray.project(&point);
                    if distance < min_distance {
                        min_distance = distance;
                        output = Some(point);
                    }
                }
                None => {}
            }
        }
        return output;
    }

    pub fn furthest_intersection(&self, ray: Ray) -> Option<Point> {
        let mut output = None;
        let mut max_distance = f32::MIN;
        for idx in 0..self.points.len() - 1 {
            let line_segment = LineSegment {
                first: self.points[idx],
                second: self.points[idx + 1],
            };
            match line_segment.intersection(ray) {
                Some(point) => {
                    let distance = ray.project(&point);
                    if distance > max_distance {
                        max_distance = distance;
                        output = Some(point);
                    }
                }
                None => {}
            }
        }
        return output;
    }
}

#[derive(Copy, Clone, Debug)]
struct Direction {
    x: f32,
    y: f32,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:.2}, {:.2}]", self.x, self.y)
    }
}

impl Direction {
    pub fn new(x: f32, y: f32) -> Self {
        let length = (x.powf(2.0) + y.powf(2.0)).sqrt();
        return Direction {
            x: x / length,
            y: y / length,
        };
    }

    pub fn rotate_right(self) -> Self {
        return Direction {
            x: self.y,
            y: -self.x,
        };
    }

    // Project the point onto the vector.
    pub fn project(&self, point: &Point) -> f32 {
        return self.x * point.x + self.y * point.y;
    }

    pub fn cosine_distance(&self, other: &Direction) -> f32 {
        return 1. - (self.x * other.x + self.y * other.y);
    }
}

impl Neg for Direction {
    type Output = Self;

    fn neg(self) -> Self::Output {
        return Direction::new(-self.x, -self.y);
    }
}

#[derive(Copy, Clone, Debug)]
struct Ray {
    start: Point,
    direction: Direction,
}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}->", self.start, self.direction)
    }
}

impl Ray {
    pub fn get_endpoint(&self, length: f32) -> Point {
        return Point::new(
            self.start.x + self.direction.x * length,
            self.start.y + self.direction.y * length,
        );
    }

    pub fn intersection(&self, other: Ray) -> Option<Point> {
        let det = self.direction.x * (-other.direction.y) - (-other.direction.x) * self.direction.y;
        if det == 0.0 {
            return None;
        }
        let t_1 = ((other.start.x - self.start.x) * (-other.direction.y)
            - (-other.direction.x) * (other.start.y - self.start.y))
            / det;
        let t_2 = (self.direction.x * (other.start.y - self.start.y)
            - (other.start.x - self.start.x) * self.direction.y)
            / det;
        if t_1 < 0.0 || t_2 < 0.0 {
            return None;
        }
        return Some(Point {
            x: self.start.x + self.direction.x * t_1,
            y: self.start.y + self.direction.y * t_1,
        });
    }

    pub fn terminate(&self, other: Ray) -> Option<(f32, f32)> {
        let det = self.direction.x * (-other.direction.y) - (-other.direction.x) * self.direction.y;
        if det == 0.0 {
            return None;
        }
        let t_1 = ((other.start.x - self.start.x) * (-other.direction.y)
            - (-other.direction.x) * (other.start.y - self.start.y))
            / det;
        let t_2 = (self.direction.x * (other.start.y - self.start.y)
            - (other.start.x - self.start.x) * self.direction.y)
            / det;
        if t_1 < 0.0 || t_2 < 0.0 {
            return None;
        }
        return Some((t_1, t_2));
    }

    pub fn project(&self, point: &Point) -> f32 {
        let relative_point = Point {
            x: point.x - self.start.x,
            y: point.y - self.start.y,
        };
        return self.direction.project(&relative_point);
    }

    // Returns true if the point is "in front" of the ray
    pub fn in_front(&self, point: &Point) -> bool {
        return self.direction.project(point) > self.direction.project(&self.start);
    }
}

// A horizontal (left-opening) parabola whose directrix is a line x = directrix.
// x = (y - focus.y)^2 / 2 / (focus.x - directrix) + (focus.x + directrix) / 2

// ray
// direction.x * (y - start.y) = direction.y * (x - start.x)
// x = direction.x / direction.y * (y - start.y) + start.x

// intersection
// (1/2/(focus.x - directrix)) y^2 + (-focus.y/ (focus.x - directrix) - direction.x / direction.y) y + focus.y^2 / 2 / (focus.x - directrix)+ (focus.x - directrix) / 2 + direction.x / direction.y * start.y - start.x = 0

#[derive(Debug)]
struct Parabola {
    focus: Point,
    directrix: f32,
}

impl Parabola {
    fn x_from_y(&self, y: f32) -> f32 {
        return (y - self.focus.y).powf(2.) / 2. / (self.focus.x - self.directrix)
            + (self.focus.x + self.directrix) / 2.;
    }

    pub fn intersection(&self, ray: &Ray) -> Option<Point> {
        if ray.direction.y == 0.0 {
            // the ray is perpendicular to the parabola's directrix
            // there is at most one intersection
            let y = ray.start.y;
            let x = self.x_from_y(y);
            let point = Point { x: x, y: y };
            if !ray.in_front(&point) {
                // the ray starts inside the parabola
                // there is no intersection
                return None;
            }
            return Some(point);
        }
        let a = 1. / 2. / (self.focus.x - self.directrix);
        let b = -self.focus.y / (self.focus.x - self.directrix) - ray.direction.x / ray.direction.y;
        let c = self.focus.y.powf(2.0) / 2. / (self.focus.x - self.directrix)
            + (self.focus.x + self.directrix) / 2.
            + ray.direction.x / ray.direction.y * ray.start.y
            - ray.start.x;
        let d = b.powf(2.) - 4. * a * c;
        if d == 0.0 {
            // the ray is tangent to the parabola
            // there is at most one intersection
            let y = -b / (2. * a);
            let x = self.x_from_y(y);
            let point = Point { x: x, y: y };
            if !ray.in_front(&point) {
                // the ray starts inside the parabola
                // there is no intersection
                return None;
            }
            return Some(point);
        }
        if d < 0.0 {
            // there are no intersections
            return None;
        }
        // there are two intersections of the ray's line with the parabola
        let y_plus = (-b + d.sqrt()) / (2. * a);
        let y_minus = (-b - d.sqrt()) / (2. * a);
        let x_plus = self.x_from_y(y_plus);
        let x_minus = self.x_from_y(y_minus);
        let options = vec![
            Point {
                x: x_plus,
                y: y_plus,
            },
            Point {
                x: x_minus,
                y: y_minus,
            },
        ];

        // remove intersections "behind" the ray's origin
        let mut valid_options: Vec<Point> = options
            .into_iter()
            .filter(|ixn| ray.in_front(ixn))
            .collect();
        if valid_options.len() == 0 {
            // there are no intersections "in front" of the ray's origin
            return None;
        }

        // find the intersection closest to the ray's origin
        valid_options.sort_by(|a, b| {
            a.distance_from(&ray.start)
                .total_cmp(&b.distance_from(&ray.start))
        });
        return Some(valid_options[0]);
    }

    pub fn tangent_at(&self, point: &Point) -> Direction {
        // x = (y - focus.y)^2 / 2 / (focus.x - directrix) + (focus.x + directrix) / 2
        // dx/dy = (y - focus.y) / (focus.x - directrix)

        // ASSUME that the point is on the parabola!
        let dxdy = (point.y - self.focus.y) / (self.focus.x - self.directrix);
        return Direction::new(dxdy, 1.);
    }

    pub fn normal(&self, point: &Point) -> Direction {
        let tangent = self.tangent_at(point);
        return tangent.rotate_right();
    }

    pub fn from_arc(arc: &Arc, directrix: f32) -> Self {
        return Parabola {
            focus: arc.focus,
            directrix: directrix,
        };
    }
}

#[derive(Debug)]
struct Slot {
    id: u32,
    value: Node,
    lower_neighbor: Option<u32>,
    upper_neighbor: Option<u32>,
}

static GLOBAL_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

pub fn generate_id() -> u32 {
    GLOBAL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Slot {
    pub fn seq_str(&self, beachline: &Beachline) -> String {
        let upper_str = match self.upper_neighbor {
            Some(upper_neighbor_id) => {
                "-".to_owned() + &beachline.nodes[&upper_neighbor_id].seq_str_up(beachline)
            }
            None => "".to_string(),
        };
        return format!("root {}; ", self.id) + &self.seq_str_down(beachline) + &upper_str;
    }
    pub fn seq_str_down(&self, beachline: &Beachline) -> String {
        match self.lower_neighbor {
            Some(lower_neighbor_id) => {
                let lower_neighbor = &beachline.nodes[&lower_neighbor_id];
                return format!(
                    "{}-{}",
                    lower_neighbor.seq_str_down(beachline),
                    format!("{}", self.id)
                );
            }
            None => {
                return format!("{}", format!("{}", self.id));
            }
        }
    }
    pub fn seq_str_up(&self, beachline: &Beachline) -> String {
        match self.upper_neighbor {
            Some(upper_neighbor_id) => {
                if upper_neighbor_id == self.id {
                    panic!("This is bad.")
                }
                let upper_neighbor = &beachline.nodes[&upper_neighbor_id];
                return format!(
                    "{}-{}",
                    format!("{}", self.id),
                    upper_neighbor.seq_str_up(beachline)
                );
            }
            None => {
                return format!("{}", format!("{}", self.id));
            }
        }
    }
    pub fn str(&self, beachline: &Beachline, prefix: String) -> String {
        match self.value {
            Node::Arc(arc) => {
                return format!("a{:03} {}", self.id, arc.focus);
            }
            Node::Edge(edge) => {
                let first = format!(
                    "e{:03}{} ─┬─ {}",
                    self.id,
                    ":",
                    beachline.nodes[&edge.upper_child]
                        .str(beachline, prefix.clone() + "       │  "),
                );
                let second = format!(
                    "       └─ {}",
                    beachline.nodes[&edge.lower_child]
                        .str(beachline, prefix.clone() + "          "),
                );
                return first + "\n" + &prefix + &second;
            }
        }
    }

    pub fn builder() -> SlotBuilder {
        return SlotBuilder::new();
    }

    pub fn is_leaf(&self) -> bool {
        match self.value {
            Node::Arc(_) => true,
            _ => false,
        }
    }
}

struct SlotBuilder {
    id: u32,
    value: Option<Node>,
    lower_neighbor: Option<u32>,
    upper_neighbor: Option<u32>,
}

impl SlotBuilder {
    pub fn new() -> Self {
        return SlotBuilder {
            id: generate_id(),
            value: None,
            lower_neighbor: None,
            upper_neighbor: None,
        };
    }

    pub fn build(&self) -> Slot {
        let value = self.value.unwrap();
        return Slot {
            id: self.id,
            value: value,
            lower_neighbor: self.lower_neighbor,
            upper_neighbor: self.upper_neighbor,
        };
    }
}

#[derive(Copy, Clone, Debug)]
enum Node {
    Arc(Arc),
    Edge(Edge),
}

impl Node {
    pub fn get_arc(self) -> Option<Arc> {
        return match self {
            Node::Arc(arc) => Some(arc),
            Node::Edge(_) => None,
        };
    }

    pub fn get_edge(self) -> Option<Edge> {
        return match self {
            Node::Edge(edge) => Some(edge),
            Node::Arc(_) => None,
        };
    }
}

#[derive(Copy, Clone, Debug)]
struct Edge {
    ray: Ray,
    lower_child: u32,
    upper_child: u32,
    parent: Option<u32>,
    length: Option<f32>, // present if the edge is complete
}

#[derive(Copy, Clone, Debug)]
struct Arc {
    focus: Point,
    parent: Option<u32>,
}

impl Arc {
    pub fn new(focus: Point, parent: Option<u32>) -> Self {
        return Arc {
            focus: focus,
            parent: parent,
        };
    }

    pub fn intersection(&self, ray: &Ray, directrix: f32) -> Option<Point> {
        return self.get_parabola(directrix).intersection(ray);
    }

    pub fn get_parabola(&self, directrix: f32) -> Parabola {
        return Parabola::from_arc(self, directrix);
    }
}

#[derive(Debug)]
struct Beachline {
    root: Option<u32>,
    nodes: HashMap<u32, Slot>,
    complete_edges: Vec<Edge>,
    complete_sites: Vec<Point>,
}

impl fmt::Display for Beachline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.root {
            Some(root_node_id) => {
                let root_slot = &self.nodes[&root_node_id];
                write!(
                    f,
                    "\n{}\n{}",
                    root_slot.str(self, "".to_string()),
                    root_slot.seq_str(&self),
                )
            }
            None => {
                write!(f, "empty Beachline")
            }
        }
    }
}

impl Beachline {
    pub fn new() -> Self {
        return Beachline {
            root: None,
            nodes: HashMap::new(),
            complete_edges: vec![],
            complete_sites: vec![],
        };
    }

    // each edge corresponds to a y value where two arcs collide
    pub fn remove_arc(&mut self, target_arc_slot_id: u32, directrix: f32) -> Vec<u32> {
        if !self.nodes.contains_key(&target_arc_slot_id) {
            return vec![];
        }
        let target_arc_slot = &self.nodes[&target_arc_slot_id];
        let target_arc = target_arc_slot.value.get_arc().unwrap();
        if !matches!(target_arc_slot.value, Node::Arc(_)) {
            // just pass?
            return vec![];
        }
        // for a circle event, edges should exist on both sides
        let lower_edge_slot = &self.nodes[&target_arc_slot.lower_neighbor.unwrap()];
        let upper_edge_slot = &self.nodes[&target_arc_slot.upper_neighbor.unwrap()];
        let lower_edge_slot_id = lower_edge_slot.id;
        let upper_edge_slot_id = upper_edge_slot.id;
        let lower_arc_slot = &self.nodes[&lower_edge_slot.lower_neighbor.unwrap()];
        let upper_arc_slot = &self.nodes[&upper_edge_slot.upper_neighbor.unwrap()];
        let lower_arc_slot_id = lower_arc_slot.id;
        let upper_arc_slot_id = upper_arc_slot.id;
        let lower_arc = lower_arc_slot.value.get_arc().unwrap();
        let lower_site = lower_arc.focus;
        let upper_site = upper_arc_slot.value.get_arc().unwrap().focus;
        let lower_edge = &lower_edge_slot.value.get_edge().unwrap();
        let upper_edge = &upper_edge_slot.value.get_edge().unwrap();
        let new_start = lower_edge.ray.intersection(upper_edge.ray).unwrap();
        let mut new_direction = lower_site.perpendicular(&upper_site);
        let normal = lower_arc.get_parabola(directrix).normal(&new_start);
        let projection = normal.project(&Point {
            x: new_direction.x,
            y: new_direction.y,
        });
        if projection < 0.0 {
            new_direction = -new_direction;
        }
        let new_ray = Ray {
            start: new_start,
            direction: new_direction,
        };
        // adjust binary tree
        if lower_edge.upper_child == target_arc_slot_id {
            // connect lower_edge.lower_child to lower_edge.parent
            // this cuts out lower_edge and its upper_child (AKA target_arc)
            let new_child_slot_id = lower_edge.lower_child;
            match lower_edge.parent {
                Some(parent_slot_id) => {
                    let parent_slot = self.nodes.get_mut(&parent_slot_id).unwrap();
                    let mut parent_edge = parent_slot.value.get_edge().unwrap();
                    if parent_edge.lower_child == lower_edge_slot_id {
                        parent_edge.lower_child = new_child_slot_id;
                    } else if parent_edge.upper_child == lower_edge_slot_id {
                        parent_edge.upper_child = new_child_slot_id;
                    }
                    // reassign edge to slot
                    parent_slot.value = Node::Edge(parent_edge);
                }
                None => {}
            }
            let new_child_slot = self.nodes.get_mut(&new_child_slot_id).unwrap();
            match new_child_slot.value {
                Node::Arc(mut new_child_mut) => {
                    new_child_mut.parent = lower_edge.parent;
                    new_child_slot.value = Node::Arc(new_child_mut);
                }
                Node::Edge(mut new_child_mut) => {
                    new_child_mut.parent = lower_edge.parent;
                    new_child_slot.value = Node::Edge(new_child_mut);
                }
            }
            // fetch upper_edge anew because we've modified self.nodes
            let target_arc_slot = &self.nodes[&target_arc_slot_id];
            let upper_edge_slot = &self.nodes[&target_arc_slot.upper_neighbor.unwrap()];
            let upper_edge = &upper_edge_slot.value.get_edge().unwrap();
            // create a new_edge between arcs lower_neighbor.lower_neighbor and upper_neighbor.upper_neighbor
            let new_edge = Edge {
                ray: new_ray,
                lower_child: upper_edge.lower_child,
                upper_child: upper_edge.upper_child,
                parent: upper_edge.parent,
                length: None,
            };
            // replace upper_edge with new_edge
            let mut new_edge_slot_builder = Slot::builder();
            new_edge_slot_builder.lower_neighbor = Some(lower_arc_slot_id);
            new_edge_slot_builder.upper_neighbor = Some(upper_arc_slot_id);
            new_edge_slot_builder.value = Some(Node::Edge(new_edge));

            match &mut self.nodes.get_mut(&upper_edge.lower_child).unwrap().value {
                Node::Arc(arc) => arc.parent = Some(new_edge_slot_builder.id),
                Node::Edge(edge) => edge.parent = Some(new_edge_slot_builder.id),
            };

            match &mut self.nodes.get_mut(&upper_edge.upper_child).unwrap().value {
                Node::Arc(arc) => arc.parent = Some(new_edge_slot_builder.id),
                Node::Edge(edge) => edge.parent = Some(new_edge_slot_builder.id),
            };

            match upper_edge.parent {
                Some(parent_id) => {
                    let parent_slot = self.nodes.get_mut(&parent_id).unwrap();
                    let mut parent_edge = parent_slot.value.get_edge().unwrap();
                    if parent_edge.lower_child == upper_edge_slot_id {
                        parent_edge.lower_child = new_edge_slot_builder.id;
                    } else if parent_edge.upper_child == upper_edge_slot_id {
                        parent_edge.upper_child = new_edge_slot_builder.id;
                    } else {
                        panic!("This should not be.")
                    }
                    parent_slot.value = Node::Edge(parent_edge);
                }
                None => self.root = Some(new_edge_slot_builder.id),
            }
            self.nodes
                .insert(new_edge_slot_builder.id, new_edge_slot_builder.build());
            // fix neighbors
            let lower_arc_slot_mut = self.nodes.get_mut(&lower_arc_slot_id).unwrap();
            lower_arc_slot_mut.upper_neighbor = Some(new_edge_slot_builder.id);
            let upper_arc_slot_mut = self.nodes.get_mut(&upper_arc_slot_id).unwrap();
            upper_arc_slot_mut.lower_neighbor = Some(new_edge_slot_builder.id);
        } else if upper_edge.lower_child == target_arc_slot_id {
            // connect upper_edge.upper_child to upper_edge.parent
            // this cuts out upper_edge and its lower_child
            let new_child_slot_id = upper_edge.upper_child;
            match upper_edge.parent {
                Some(parent_slot_id) => {
                    let parent_slot = self.nodes.get_mut(&parent_slot_id).unwrap();
                    let mut parent_edge = parent_slot.value.get_edge().unwrap();
                    if parent_edge.lower_child == upper_edge_slot_id {
                        parent_edge.lower_child = new_child_slot_id;
                    } else if parent_edge.upper_child == upper_edge_slot_id {
                        parent_edge.upper_child = new_child_slot_id;
                    } else {
                        panic!("This should not be.")
                    }
                    // reassign edge to slot
                    parent_slot.value = Node::Edge(parent_edge);
                }
                None => {
                    // TODO: what if upper_edge has no parent?
                }
            }
            let new_child_slot = self.nodes.get_mut(&new_child_slot_id).unwrap();
            match new_child_slot.value {
                Node::Arc(mut new_child_mut) => {
                    new_child_mut.parent = upper_edge.parent;
                    new_child_slot.value = Node::Arc(new_child_mut);
                }
                Node::Edge(mut new_child_mut) => {
                    new_child_mut.parent = upper_edge.parent;
                    new_child_slot.value = Node::Edge(new_child_mut);
                }
            }
            // fetch lower_edge anew because we've modified self.nodes
            let target_arc_slot = &self.nodes[&target_arc_slot_id];
            let lower_edge_slot = &self.nodes[&target_arc_slot.lower_neighbor.unwrap()];
            let lower_edge = &lower_edge_slot.value.get_edge().unwrap();
            // create a new_edge between arcs lower_neighbor.lower_neighbor and upper_neighbor.upper_neighbor
            let new_edge = Edge {
                ray: new_ray,
                lower_child: lower_edge.lower_child,
                upper_child: lower_edge.upper_child,
                parent: lower_edge.parent,
                length: None,
            };

            // replace lower_edge with new_edge
            let mut new_edge_slot_builder = Slot::builder();
            new_edge_slot_builder.lower_neighbor = Some(lower_arc_slot_id);
            new_edge_slot_builder.upper_neighbor = Some(upper_arc_slot_id);
            new_edge_slot_builder.value = Some(Node::Edge(new_edge));

            match &mut self.nodes.get_mut(&lower_edge.lower_child).unwrap().value {
                Node::Arc(arc) => arc.parent = Some(new_edge_slot_builder.id),
                Node::Edge(edge) => edge.parent = Some(new_edge_slot_builder.id),
            };

            match &mut self.nodes.get_mut(&lower_edge.upper_child).unwrap().value {
                Node::Arc(arc) => arc.parent = Some(new_edge_slot_builder.id),
                Node::Edge(edge) => edge.parent = Some(new_edge_slot_builder.id),
            };

            match lower_edge.parent {
                Some(parent_id) => {
                    let parent_slot = self.nodes.get_mut(&parent_id).unwrap();
                    let mut parent_edge = parent_slot.value.get_edge().unwrap();
                    if parent_edge.lower_child == lower_edge_slot_id {
                        parent_edge.lower_child = new_edge_slot_builder.id;
                    } else if parent_edge.upper_child == lower_edge_slot_id {
                        parent_edge.upper_child = new_edge_slot_builder.id;
                    } else {
                        panic!("This should not be.")
                    }
                    parent_slot.value = Node::Edge(parent_edge);
                }
                None => self.root = Some(new_edge_slot_builder.id),
            }
            self.nodes
                .insert(new_edge_slot_builder.id, new_edge_slot_builder.build());

            // fix neighbors
            let lower_arc_slot_mut = self.nodes.get_mut(&lower_arc_slot_id).unwrap();
            lower_arc_slot_mut.upper_neighbor = Some(new_edge_slot_builder.id);
            let upper_arc_slot_mut = self.nodes.get_mut(&upper_arc_slot_id).unwrap();
            upper_arc_slot_mut.lower_neighbor = Some(new_edge_slot_builder.id);
        }
        let mut arcs_to_check = vec![];
        match self.nodes[&lower_edge_slot_id].lower_neighbor {
            Some(lower_arc_slot_id) => {
                arcs_to_check.push(lower_arc_slot_id);
            }
            None => {}
        }
        match self.nodes[&upper_edge_slot_id].upper_neighbor {
            Some(upper_arc_slot_id) => {
                arcs_to_check.push(upper_arc_slot_id);
            }
            None => {}
        }
        // remove obsolete edges
        self.nodes.remove(&lower_edge_slot_id);
        self.nodes.remove(&upper_edge_slot_id);
        // remove obsolete node
        self.nodes.remove(&target_arc_slot_id);

        self.complete_sites.push(target_arc.focus);

        let mut my_lower_edge = *lower_edge;
        let mut my_upper_edge = *upper_edge;
        let (lower_length, upper_length) = my_lower_edge.ray.terminate(my_upper_edge.ray).unwrap();
        my_lower_edge.length = Some(lower_length);
        my_upper_edge.length = Some(upper_length);
        self.complete_edges.push(my_lower_edge);
        self.complete_edges.push(my_upper_edge);
        return arcs_to_check;
    }

    fn get_lower_edge(&self, arc_slot_id: u32) -> Option<Edge> {
        let target_arc_slot = &self.nodes[&arc_slot_id];
        match target_arc_slot.lower_neighbor {
            Some(id) => {
                let lower_edge_slot = &self.nodes[&id];
                return Some(lower_edge_slot.value.get_edge().unwrap());
            }
            None => {
                return None;
            }
        }
    }

    fn get_upper_edge(&self, arc_slot_id: u32) -> Option<Edge> {
        let target_arc_slot = &self.nodes[&arc_slot_id];
        match target_arc_slot.upper_neighbor {
            Some(id) => {
                let upper_edge_slot = &self.nodes[&id];
                return Some(upper_edge_slot.value.get_edge().unwrap());
            }
            None => {
                return None;
            }
        }
    }

    // check for new circle events
    // for each of the neighboring arcs,
    pub fn check_for_circle_events(&self, arc_slot_id: u32) -> Option<f32> {
        // Returns the x location of the directrix when the arc is squashed,
        // or None if it is not.
        let lower_edge = match self.get_lower_edge(arc_slot_id) {
            Some(edge) => edge,
            None => return None,
        };
        let upper_edge = match self.get_upper_edge(arc_slot_id) {
            Some(edge) => edge,
            None => return None,
        };
        // if the arc's bounding edge-rays intersect, it will be squashed
        match lower_edge.ray.intersection(upper_edge.ray) {
            Some(intersection) => {
                // given d the distance between the arc focus and the intersection
                let d = intersection
                    .distance_from(&self.nodes[&arc_slot_id].value.get_arc().unwrap().focus);
                // the arc should be removed and the edges terminated when the directrix is distance d from the intersection
                return Some(intersection.x + d);
            }
            None => {
                return None;
            }
        }
    }

    pub fn add_site(&mut self, site: &Site) -> Vec<u32> {
        if self.root.is_none() {
            let mut new_arc_slot_builder = Slot::builder();
            let new_arc = Arc::new(site.location, None);
            new_arc_slot_builder.value = Some(Node::Arc(new_arc));
            let new_arc_slot = new_arc_slot_builder.build();
            self.root = Some(new_arc_slot.id);
            self.nodes.insert(new_arc_slot.id, new_arc_slot);
            return vec![];
        }
        // get arc and slot that will be replaced with new subtree
        let target_slot_id = self.get_slot_id_at(&self.nodes[&self.root.unwrap()], &site.location);
        let target_slot = self.nodes.get_mut(&target_slot_id).unwrap();
        let target_arc = target_slot.value.get_arc().unwrap();

        // geometry
        let ray = Ray {
            start: site.location,
            direction: Direction::new(-1.0, 0.0),
        };
        let site_to_arc_ray_start = target_arc.intersection(&ray, site.location.x).unwrap();
        let parabola = target_arc.get_parabola(site.location.x);
        let up_tangent = parabola.tangent_at(&site_to_arc_ray_start);
        let down_tangent = -up_tangent;
        let up_ray = Ray {
            start: site_to_arc_ray_start,
            direction: up_tangent,
        };
        let down_ray = Ray {
            start: site_to_arc_ray_start,
            direction: down_tangent,
        };

        // create all slot builders; assigns IDs
        let mut bottom_arc_slot_builder = Slot::builder();
        let mut new_arc_slot_builder = Slot::builder();
        let mut top_arc_slot_builder = Slot::builder();
        let mut bottom_edge_slot_builder = Slot::builder();

        // create new arcs
        let bottom_arc = Arc::new(target_arc.focus, Some(bottom_edge_slot_builder.id));
        bottom_arc_slot_builder.value = Some(Node::Arc(bottom_arc));
        let new_arc = Arc::new(site.location, Some(bottom_edge_slot_builder.id));
        new_arc_slot_builder.value = Some(Node::Arc(new_arc));
        let top_arc = Arc::new(target_arc.focus, Some(target_slot_id));
        top_arc_slot_builder.value = Some(Node::Arc(top_arc));

        // create new edges
        let bottom_edge = Edge {
            ray: down_ray,
            lower_child: bottom_arc_slot_builder.id,
            upper_child: new_arc_slot_builder.id,
            parent: Some(target_slot_id),
            length: None,
        };
        bottom_edge_slot_builder.value = Some(Node::Edge(bottom_edge));
        let top_edge = Edge {
            ray: up_ray,
            lower_child: bottom_edge_slot_builder.id,
            upper_child: top_arc_slot_builder.id,
            parent: target_arc.parent,
            length: None,
        };

        let target_slot_upper_neighbor_id = target_slot.upper_neighbor;
        let target_slot_lower_neighbor_id = target_slot.lower_neighbor;
        // assign neighbors
        top_arc_slot_builder.upper_neighbor = target_slot_upper_neighbor_id;
        top_arc_slot_builder.lower_neighbor = Some(target_slot_id);
        // target slot becomes "top edge"
        target_slot.upper_neighbor = Some(top_arc_slot_builder.id);
        target_slot.lower_neighbor = Some(new_arc_slot_builder.id);
        new_arc_slot_builder.upper_neighbor = Some(target_slot_id);
        new_arc_slot_builder.lower_neighbor = Some(bottom_edge_slot_builder.id);
        bottom_edge_slot_builder.upper_neighbor = Some(new_arc_slot_builder.id);
        bottom_edge_slot_builder.lower_neighbor = Some(bottom_arc_slot_builder.id);
        bottom_arc_slot_builder.upper_neighbor = Some(bottom_edge_slot_builder.id);
        bottom_arc_slot_builder.lower_neighbor = target_slot_lower_neighbor_id;

        // attach new subtree
        target_slot.value = Node::Edge(top_edge);
        self.nodes
            .insert(top_arc_slot_builder.id, top_arc_slot_builder.build());
        self.nodes
            .insert(new_arc_slot_builder.id, new_arc_slot_builder.build());
        self.nodes.insert(
            bottom_edge_slot_builder.id,
            bottom_edge_slot_builder.build(),
        );
        self.nodes
            .insert(bottom_arc_slot_builder.id, bottom_arc_slot_builder.build());

        // fix neighbor links around inserted subtree
        match target_slot_upper_neighbor_id {
            Some(id) => {
                let upper_neighbor = self.nodes.get_mut(&id).unwrap();
                upper_neighbor.lower_neighbor = Some(top_arc_slot_builder.id);
            }
            None => {}
        }
        match target_slot_lower_neighbor_id {
            Some(id) => {
                let lower_neighbor = self.nodes.get_mut(&id).unwrap();
                lower_neighbor.upper_neighbor = Some(bottom_arc_slot_builder.id);
            }
            None => {}
        }
        return vec![bottom_arc_slot_builder.id, top_arc_slot_builder.id];
    }

    pub fn get_endpoint_y(&self, edge_slot: &Slot, directrix: f32) -> f32 {
        let lower_arc = self.nodes[&edge_slot.lower_neighbor.unwrap()]
            .value
            .get_arc()
            .unwrap();
        let edge = edge_slot.value.get_edge().unwrap();
        return lower_arc.intersection(&edge.ray, directrix).unwrap().y;
    }

    pub fn get_slot_id_at(&self, slot: &Slot, site: &Point) -> u32 {
        if slot.is_leaf() {
            return slot.id;
        }
        let directrix = site.x;
        let edge = slot.value.get_edge().unwrap();
        if self.get_endpoint_y(slot, directrix) >= site.y {
            return self.get_slot_id_at(&self.nodes[&edge.lower_child], site);
        } else {
            return self.get_slot_id_at(&self.nodes[&edge.upper_child], site);
        }
    }

    pub fn handle_site_event(&mut self, site: Site) -> Vec<Event> {
        println!("handling site event: {:?}", site);
        let mut events = vec![];
        let arc_slot_ids = self.add_site(&site);
        for arc_slot_id in arc_slot_ids {
            let circle_event = self.check_for_circle_events(arc_slot_id);
            println!("adding circle event: {:?}", circle_event);
            match circle_event {
                Some(directrix) => {
                    events.push(Event::Circle(CircleEvent {
                        arc_slot_id: arc_slot_id,
                        directrix: directrix,
                    }));
                }
                None => {}
            }
        }
        return events;
    }

    pub fn handle_circle_event(&mut self, circle_event: CircleEvent) -> Vec<Event> {
        println!("handling circle event");
        let mut events = vec![];
        let arc_slot_ids = self.remove_arc(circle_event.arc_slot_id, circle_event.directrix);
        for arc_slot_id in arc_slot_ids {
            let circle_event = self.check_for_circle_events(arc_slot_id);
            println!("adding circle event: {:?}", circle_event);
            match circle_event {
                Some(directrix) => {
                    events.push(Event::Circle(CircleEvent {
                        arc_slot_id: arc_slot_id,
                        directrix: directrix,
                    }));
                }
                None => {}
            }
        }
        return events;
    }
}

#[derive(Debug, Clone, Copy)]
struct CircleEvent {
    arc_slot_id: u32,
    directrix: f32,
}

#[derive(Debug, Clone, Copy)]
enum Event {
    Site(Site),
    Circle(CircleEvent),
}

impl Event {
    pub fn get_position(self) -> f32 {
        return match self {
            Event::Site(site) => site.location.x,
            Event::Circle(circle_event) => circle_event.directrix,
        };
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.get_position() == other.get_position()
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // Compare based on x only, and backwards (low x is "higher")
        other.get_position().total_cmp(&self.get_position())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Site {
    location: Point,
}

impl PartialEq for Site {
    fn eq(&self, other: &Self) -> bool {
        self.location.x == other.location.x
    }
}

impl Eq for Site {}

fn add_sites(points: Vec<(f32, f32)>) -> HashMap<usize, Site> {
    let mut sites = HashMap::<usize, Site>::new();

    for (idx, point) in points.iter().enumerate() {
        sites.insert(
            idx,
            Site {
                location: Point {
                    x: point.0,
                    y: point.1,
                },
            },
        );
    }

    return sites;
}

fn fortunes(sites: HashMap<usize, Site>, boundaries: &Polyline) -> Beachline {
    let mut events = BinaryHeap::new();

    for idx in 0..sites.len() {
        events.push(Event::Site(sites[&idx]));
    }

    let mut beachline = Beachline::new();

    while let Some(event) = events.pop() {
        match event {
            Event::Site(site) => {
                for event in beachline.handle_site_event(site) {
                    events.push(event);
                }
            }
            Event::Circle(circle_event) => {
                for event in beachline.handle_circle_event(circle_event) {
                    events.push(event);
                }
            }
        }
        println!("beachline: {}", beachline);
        // println!("complete edges: {:?}", beachline.complete_edges);
    }

    // complete rays
    for node in beachline.nodes.values() {
        match node.value {
            Node::Arc(_) => {}
            Node::Edge(mut edge) => match boundaries.furthest_intersection(edge.ray) {
                Some(point) => {
                    edge.length = Some(edge.ray.project(&point));
                    beachline.complete_edges.push(edge);
                }
                None => {
                    println!("found no intersection with boundaries")
                }
            },
        }
    }

    return beachline;
}

fn get_paths_from_edge(edge: &Edge, color: &str) -> Vec<Path> {
    let start = edge.ray.start;
    let start_plus = Point {
        x: start.x + edge.ray.direction.x * 0.1,
        y: start.y + edge.ray.direction.y * 0.1,
    };
    let end = Point {
        x: start.x + edge.ray.direction.x * edge.length.unwrap(),
        y: start.y + edge.ray.direction.y * edge.length.unwrap(),
    };
    return vec![
        get_path_from_points(&start, &end, color),
        get_path_from_points(&start, &start_plus, "red"),
    ];
}

fn get_path_from_points(start: &Point, end: &Point, color: &str) -> Path {
    let data = Data::new()
        .move_to((start.x, start.y))
        .line_by((end.x - start.x, end.y - start.y))
        .close();

    return Path::new()
        .set("fill", "none")
        .set("stroke", color)
        .set("stroke-width", 0.1)
        .set("opacity", 0.3)
        .set("d", data);
}

fn plot(beachline: &Beachline, boundaries: &Polyline) {
    let mut document = Document::new().set("viewBox", (-2, -2, 4, 4));

    for idx in 0..boundaries.points.len() - 1 {
        let start = boundaries.points[idx];
        let end = boundaries.points[idx + 1];
        let data = Data::new()
            .move_to((start.x, start.y))
            .line_by((end.x - start.x, end.y - start.y))
            .close();

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 0.1)
            .set("d", data);

        document = document.add(path);
    }

    for node in beachline.nodes.values() {
        match node.value {
            Node::Arc(arc) => {
                let path = Circle::new()
                    .set("fill", "black")
                    .set("opacity", 0.3)
                    .set("cx", arc.focus.x)
                    .set("cy", arc.focus.y)
                    .set("r", 0.1);

                document = document.add(path);
            }
            Node::Edge(_) => {}
        }
    }

    for edge in &beachline.complete_edges {
        for path in get_paths_from_edge(&edge, "green") {
            document = document.add(path);
        }
    }

    for point in &beachline.complete_sites {
        let path = Circle::new()
            .set("fill", "blue")
            .set("opacity", 0.3)
            .set("cx", point.x)
            .set("cy", point.y)
            .set("r", 0.1);

        document = document.add(path);
    }

    svg::save("image.svg", &document).unwrap();
}

pub fn run_fortunes() {
    let mut rng = rand::rng();

    let mut sites = HashMap::<usize, Site>::new();

    // random
    let num_sites = 20;
    for idx in 0..num_sites {
        sites.insert(
            idx,
            Site {
                location: Point {
                    x: rng.random::<f32>() * 4. - 2.,
                    y: rng.random::<f32>() * 4. - 2.,
                },
            },
        );
    }

    // // fail bug 8
    // let points = vec![
    //     (-1.26, 1.02),
    //     (0.36, 0.64),
    //     (0.83, 0.45),
    //     (-1.07, -1.07),
    //     (1.74, -1.05),
    // ];
    // sites = add_sites(points);

    // // fail bug 7
    // let points = vec![
    //     (-1.77, 1.60),
    //     (0.24, 1.60),
    //     (1.14, -1.51),
    //     (-1.20, -0.76),
    //     (0.54, -1.08),
    // ];
    // sites = add_sites(points);

    // // fail bug 6
    // let points = vec![
    //     (-1.00, 1.97),
    //     (1.68, 0.86),
    //     (-0.92, -0.36),
    //     (0.98, -1.07),
    //     (0.32, -1.18),
    // ];
    // sites = add_sites(points);

    // // fail bug 5
    // let points = vec![(-0.46, -0.94), (-0.28, 1.74), (-0.35, -0.10), (1.00, -1.10)];
    // sites = add_sites(points);

    // // fail bug 4
    // let points = vec![(-1.68, 0.22), (-1.06, -0.80), (-0.25, -0.40), (1.57, -1.90)];
    // sites = add_sites(points);

    // // // fail bug 3
    // let points = vec![(-0.10, 1.57), (0.62, 1.48), (1.66, 0.43), (1.84, -0.19)];
    // sites = add_sites(points);

    // // fail bug 2
    // let points = vec![
    //     (-1.77, 1.17),
    //     (-1.76, 1.89),
    //     (-1.57, -1.87),
    //     (1.70, 0.44),
    //     (0.32, -1.18),
    // ];
    // sites = add_sites(points);

    // // fail bug 1
    // let points = vec![(0.14, -1.10), (0.44, -0.08), (0.42, -1.09)];
    // sites = add_sites(points);

    // // // edge bug 2
    // let points = vec![(0.00, 0.20), (0.50, 1.50), (0.70, 0.10)];
    // sites = add_sites(points);

    // // in a line at y=0
    // let points = vec![(0.00, 0.00), (1.00, 0.00), (2.00, 0.00)];
    // sites = add_sites(points);

    // // in a right-opening v
    // let points = vec![(0.00, 0.00), (1.00, 1.00), (1.01, -1.00)];
    // sites = add_sites(points);

    // // edge bug 1
    // let points = vec![(-1.90, 1.00), (1.00, 0.00), (1.80, 1.30)];
    // sites = add_sites(points);

    let mut boundaries = Polyline::new();
    boundaries.points.push(Point { x: -2.0, y: -2.0 });
    boundaries.points.push(Point { x: -2.0, y: 2.0 });
    boundaries.points.push(Point { x: 2.0, y: 2.0 });
    boundaries.points.push(Point { x: 2.0, y: -2.0 });
    boundaries.points.push(Point { x: -2.0, y: -2.0 });

    let beachline = fortunes(sites, &boundaries);
    plot(&beachline, &boundaries);
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_fortunes() {
        run_fortunes();
        // assert_eq!(fortunes(), HashMap::new());
    }

    #[test]
    fn test_ray() {
        let ray1 = Ray {
            start: Point::new(0.0, 0.0),
            direction: Direction::new(1.0, 0.0),
        };
        let ray2 = Ray {
            start: Point::new(1.0, 1.0),
            direction: Direction::new(0.0, -1.0),
        };
        let expected_intersection = Point::new(1.0, 0.0);
        assert!(
            ray1.intersection(ray2)
                .unwrap()
                .distance_from(&expected_intersection)
                < 1e-5
        );

        let ray1 = Ray {
            start: Point::new(0.0, 0.0),
            direction: Direction::new(1.0, 1.0),
        };
        let ray2 = Ray {
            start: Point::new(3.0, 0.0),
            direction: Direction::new(0.0, 1.0),
        };
        let expected_intersection = Point::new(3.0, 3.0);
        assert!(
            ray1.intersection(ray2)
                .unwrap()
                .distance_from(&expected_intersection)
                < 1e-5
        );
    }

    #[test]
    fn test_parabola_1() {
        // ray perpendicular to directrix, passing through parabola
        let focus = Point::new(2.0, 0.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 0.0,
        };
        let ray = Ray {
            start: Point::new(0.0, 0.0),
            direction: Direction::new(1.0, 0.0),
        };
        let expected_intersection = Point::new(1.0, 0.0);
        assert!(
            parabola
                .intersection(&ray)
                .unwrap()
                .distance_from(&expected_intersection)
                < 1e-5
        );
    }

    #[test]
    fn test_parabola_2() {
        // ray parallel to directrix, passing through both sides of parabola
        let focus = Point::new(2.0, 0.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 0.0,
        };
        let ray = Ray {
            start: Point::new(3.0, -10.0),
            direction: Direction::new(0.0, 1.0),
        };
        let expected_intersection = Point::new(3.0, -2.828427);
        assert!(
            parabola
                .intersection(&ray)
                .unwrap()
                .distance_from(&expected_intersection)
                < 1e-5
        );
    }

    #[test]
    fn test_parabola_3() {
        let focus = Point::new(3.0, 4.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 1.0,
        };
        let ray = Ray {
            start: Point::new(4.0, 0.0),
            direction: Direction::new(2.0, 1.0),
        };
        let expected_intersection = Point::new(5.033371, 0.516685);
        assert!(
            parabola
                .intersection(&ray)
                .unwrap()
                .distance_from(&expected_intersection)
                < 1e-5
        );
    }

    #[test]
    fn test_parabola_4() {
        let focus = Point::new(3.0, 4.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 1.0,
        };
        let ray = Ray {
            start: Point::new(4.0, 3.0),
            direction: Direction::new(2.0, 1.0),
        };
        let expected_intersection = Point::new(25.313711, 13.656855);
        assert!(
            parabola
                .intersection(&ray)
                .unwrap()
                .distance_from(&expected_intersection)
                < 1e-5
        );
    }

    #[test]
    fn test_parabola_5() {
        let focus = Point::new(2.0, 0.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 4.0,
        };
        let ray = Ray {
            start: Point::new(0.0, 1.0),
            direction: Direction::new(1.0, 0.0),
        };
        let expected_intersection = Point::new(2.75, 1.0);
        assert!(
            parabola
                .intersection(&ray)
                .unwrap()
                .distance_from(&expected_intersection)
                < 1e-5
        );
    }

    #[test]
    fn test_parabola_6() {
        let focus = Point::new(2.0, 0.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 4.0,
        };
        let ray = Ray {
            start: Point::new(6.0, 1.0),
            direction: Direction::new(0.0, 1.0),
        };
        assert!(parabola.intersection(&ray).is_none());
    }

    #[test]
    fn test_parabola_7() {
        // ray perpendicular to directrix, starting inside parabola
        let focus = Point::new(2.0, 0.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 0.0,
        };
        let ray = Ray {
            start: Point::new(3.0, 0.0),
            direction: Direction::new(1.0, 0.0),
        };
        assert!(parabola.intersection(&ray).is_none());
    }

    #[test]
    fn test_parabola_tangent_1() {
        let focus = Point::new(2.0, 0.0);
        let parabola = Parabola {
            focus: focus,
            directrix: 4.0,
        };
        let point = Point::new(2.75, 1.0);
        let expected_direction = Direction::new(-0.4472136, 0.8944272);
        println!("tangent: {:?}", parabola.tangent_at(&point));
        assert!(
            parabola
                .tangent_at(&point)
                .cosine_distance(&expected_direction)
                < 1e-5
        );
    }
}
