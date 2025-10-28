use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fmt;
use std::ops::Neg;
use svg::node::element::path::Data;
use svg::node::element::Circle;
use svg::node::element::Path;
use svg::Document;

// binary tree with ids for arcs and edges, per Fortune's algorithm
// map from arc id to adjacent edges
// map from edge id to adjacent arcs

#[derive(Copy, Clone, Debug)]
// #[derive(PartialEq)]
// #[derive(Eq)]
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
        // pointed downward
        // TODO: point right?
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
        // let first = self.first;
        // let second = self.second;
        // let fourth = ray.start;
        // let third = Point {
        //     x: ray.start.x + ray.direction.x,
        //     y: ray.start.y + ray.direction.y,
        // };
        // let den = (first.x - second.x) * (third.y - fourth.y)
        //     - (first.y - second.y) * (third.x - fourth.x);
        // let t_num =
        //     (first.x - third.x) * (third.y - fourth.y) - (first.y - third.y) * (third.x - fourth.x);
        // let u_num =
        //     (first.x - second.x) * (first.y - third.y) - (first.y - second.y) * (first.x - third.x);
        // let t = t_num / den;
        // let u = u_num / den;
        // println!("t: {:?}, u: {:?}", t, u);
        // if t < 0.0 || t > 1.0 || u < 0.0 {
        //     return None;
        // }
        // return Some(Point {
        //     x: first.x + (second.x - first.x) * t,
        //     y: first.y + (second.y - first.y) * t,
        // });
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
            println!("ray: {:?}, segment: {:?}", ray, line_segment);
            match line_segment.intersection(ray) {
                Some(point) => {
                    let distance = ray.project(&point);
                    println!("point: {:?}, distance: {:?}", point, distance);
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

pub fn generate_id() -> u32 {
    return rand::rng().random();
}

impl Slot {
    pub fn seq_str(&self, beachline: &Beachline) -> String {
        let upper_str = match self.upper_neighbor {
            Some(upper_neighbor_id) => {
                "-".to_owned() + &beachline.nodes[&upper_neighbor_id].seq_str_up(beachline)
            }
            None => "".to_string(),
        };
        return format!("root {}(u{:?}) ", self.id, self.upper_neighbor)
            + &self.seq_str_down(beachline)
            + &upper_str;
    }
    pub fn seq_str_down(&self, beachline: &Beachline) -> String {
        match self.lower_neighbor {
            Some(lower_neighbor_id) => {
                let lower_neighbor = &beachline.nodes[&lower_neighbor_id];
                return format!(
                    "{}-{}",
                    lower_neighbor.seq_str_down(beachline),
                    format!("{}(l{:?})", self.id, self.lower_neighbor)
                );
            }
            None => {
                return format!("{}", format!("{}(l{:?})", self.id, self.lower_neighbor));
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
                    format!("{}(u{:?})", self.id, self.upper_neighbor),
                    upper_neighbor.seq_str_up(beachline)
                );
            }
            None => {
                return format!("{}", format!("{}(u{:?})", self.id, self.upper_neighbor));
            }
        }
    }
    pub fn str(&self, beachline: &Beachline, prefix: String) -> String {
        match self.value {
            Node::Arc(arc) => {
                return format!("arc({}) {}", self.id, arc.focus);
            }
            Node::Edge(edge) => {
                let first = format!(
                    "edge({}l{}){} ─┬─ {}",
                    self.id,
                    edge.lower_child,
                    ":",
                    // format!(" {}", edge.ray),
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

    pub fn from_arc(arc: Arc) -> Self {
        return Self {
            id: generate_id(),
            value: Node::Arc(arc),
            lower_neighbor: None,
            upper_neighbor: None,
        };
    }

    pub fn from_edge(edge: Edge) -> Self {
        return Self {
            id: generate_id(),
            value: Node::Edge(edge),
            lower_neighbor: None,
            upper_neighbor: None,
        };
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
        };
    }

    // each edge corresponds to a y value where two arcs collide

    pub fn remove_arc(&mut self, target_arc_slot_id: u32) {
        let target_arc_slot = &self.nodes[&target_arc_slot_id];
        // for a circle event, edges should exist on both sides
        let lower_edge_slot = &self.nodes[&target_arc_slot.lower_neighbor.unwrap()];
        let upper_edge_slot = &self.nodes[&target_arc_slot.upper_neighbor.unwrap()];
        let lower_edge_slot_id = lower_edge_slot.id;
        let upper_edge_slot_id = upper_edge_slot.id;
        let lower_arc_slot = &self.nodes[&lower_edge_slot.lower_neighbor.unwrap()];
        let upper_arc_slot = &self.nodes[&upper_edge_slot.upper_neighbor.unwrap()];
        let lower_arc_slot_id = lower_arc_slot.id;
        let upper_arc_slot_id = upper_arc_slot.id;
        let lower_site = lower_arc_slot.value.get_arc().unwrap().focus;
        let upper_site = upper_arc_slot.value.get_arc().unwrap().focus;
        let lower_edge = &lower_edge_slot.value.get_edge().unwrap();
        let upper_edge = &upper_edge_slot.value.get_edge().unwrap();
        let new_start = lower_edge.ray.intersection(upper_edge.ray).unwrap();
        let new_ray = Ray {
            start: new_start,
            direction: lower_site.perpendicular(&upper_site),
        };
        // adjust binary tree
        if lower_edge.upper_child == target_arc_slot_id {
            // connect lower_edge.lower_child to lower_edge.parent
            // this cuts out lower_edge and its upper_child (AKA target_arc)
            let parent_slot = self.nodes.get_mut(&lower_edge.parent.unwrap()).unwrap(); // TODO: what if lower_edge has no parent?
            let mut parent_edge = parent_slot.value.get_edge().unwrap();
            let new_child_slot_id = lower_edge.lower_child;
            if parent_edge.lower_child == lower_edge_slot_id {
                parent_edge.lower_child = new_child_slot_id;
                // parent_slot.lower_neighbor = Some(lower_arc_slot_id);
            } else if parent_edge.upper_child == lower_edge_slot_id {
                parent_edge.upper_child = new_child_slot_id;
                // parent_slot.upper_neighbor = Some(upper_arc_slot_id);
            }
            // reassign edge to slot
            parent_slot.value = Node::Edge(parent_edge);
            match self.nodes.get_mut(&new_child_slot_id).unwrap().value {
                Node::Arc(mut new_child_mut) => {
                    new_child_mut.parent = upper_edge.parent;
                }
                Node::Edge(mut new_child_mut) => {
                    new_child_mut.parent = upper_edge.parent;
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
            let upper_edge_slot_mut = self.nodes.get_mut(&upper_edge_slot_id).unwrap();
            upper_edge_slot_mut.value = Node::Edge(new_edge);
            // fix neighbors
            upper_edge_slot_mut.upper_neighbor = Some(upper_arc_slot_id);
            upper_edge_slot_mut.lower_neighbor = Some(lower_arc_slot_id);
            let lower_arc_slot_mut = self.nodes.get_mut(&lower_arc_slot_id).unwrap();
            lower_arc_slot_mut.upper_neighbor = Some(upper_edge_slot_id);
            let upper_arc_slot_mut = self.nodes.get_mut(&upper_arc_slot_id).unwrap();
            upper_arc_slot_mut.lower_neighbor = Some(upper_edge_slot_id);
            // remove obsolete edge node
            self.nodes.remove(&lower_edge_slot_id);
        } else if upper_edge.lower_child == target_arc_slot_id {
            // connect upper_edge.upper_child to upper_edge.parent
            // this cuts out upper_edge and its lower_child
            let parent_slot = self.nodes.get_mut(&upper_edge.parent.unwrap()).unwrap(); // TODO: what if upper_edge has no parent?
            let mut parent_edge = parent_slot.value.get_edge().unwrap();
            let new_child_slot_id = upper_edge.upper_child;
            if parent_edge.lower_child == upper_edge_slot_id {
                parent_edge.lower_child = new_child_slot_id;
                // parent_slot.lower_neighbor = Some(lower_arc_slot_id);
            } else if parent_edge.upper_child == upper_edge_slot_id {
                parent_edge.upper_child = new_child_slot_id;
                // parent_slot.upper_neighbor = Some(upper_arc_slot_id);
            } else {
                panic!("This should not be.")
            }
            // reassign edge to slot (in case we copied somewhere)
            parent_slot.value = Node::Edge(parent_edge);
            match self.nodes.get_mut(&new_child_slot_id).unwrap().value {
                Node::Arc(mut new_child_mut) => {
                    new_child_mut.parent = upper_edge.parent;
                }
                Node::Edge(mut new_child_mut) => {
                    new_child_mut.parent = upper_edge.parent;
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
            let lower_edge_slot_mut = self.nodes.get_mut(&lower_edge_slot_id).unwrap();
            lower_edge_slot_mut.value = Node::Edge(new_edge);
            // fix neighbors
            lower_edge_slot_mut.upper_neighbor = Some(upper_arc_slot_id);
            lower_edge_slot_mut.lower_neighbor = Some(lower_arc_slot_id);
            let lower_arc_slot_mut = self.nodes.get_mut(&lower_arc_slot_id).unwrap();
            lower_arc_slot_mut.upper_neighbor = Some(lower_edge_slot_id);
            let upper_arc_slot_mut = self.nodes.get_mut(&upper_arc_slot_id).unwrap();
            upper_arc_slot_mut.lower_neighbor = Some(lower_edge_slot_id);
            // remove obsolete edge node
            self.nodes.remove(&upper_edge_slot_id);
        }
        let mut my_lower_edge = *lower_edge;
        let mut my_upper_edge = *upper_edge;
        let (lower_length, upper_length) = my_lower_edge.ray.terminate(my_upper_edge.ray).unwrap();
        my_lower_edge.length = Some(lower_length);
        my_upper_edge.length = Some(upper_length);
        self.complete_edges.push(my_lower_edge);
        self.complete_edges.push(my_upper_edge);
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
        // println!("target_arc: {:?}", target_arc);
        // println!("ray: {:?}", ray);
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
        let mut top_edge_slot_builder = Slot::builder();

        // create new arcs
        let bottom_arc = Arc::new(target_arc.focus, Some(bottom_edge_slot_builder.id));
        bottom_arc_slot_builder.value = Some(Node::Arc(bottom_arc));
        let new_arc = Arc::new(site.location, Some(bottom_edge_slot_builder.id));
        new_arc_slot_builder.value = Some(Node::Arc(new_arc));
        let top_arc = Arc::new(target_arc.focus, Some(top_edge_slot_builder.id));
        top_arc_slot_builder.value = Some(Node::Arc(top_arc));

        // create new edges
        let bottom_edge = Edge {
            ray: down_ray,
            lower_child: bottom_arc_slot_builder.id,
            upper_child: new_arc_slot_builder.id,
            parent: Some(top_edge_slot_builder.id),
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
        self.nodes.insert(
            bottom_edge_slot_builder.id,
            bottom_edge_slot_builder.build(),
        );
        self.nodes
            .insert(new_arc_slot_builder.id, new_arc_slot_builder.build());
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

    pub fn get_lower_arc(&self, edge: &Edge) -> &Arc {
        match &self.nodes[&edge.lower_child].value {
            Node::Edge(edge) => &self.get_lower_arc(edge),
            Node::Arc(arc) => &arc,
        }
    }
    pub fn get_upper_arc(&self, edge: &Edge) -> &Arc {
        match &self.nodes[&edge.upper_child].value {
            Node::Edge(edge) => &self.get_upper_arc(edge),
            Node::Arc(arc) => &arc,
        }
    }

    pub fn get_endpoint_y(&self, edge: &Edge, directrix: f32) -> f32 {
        let lower_arc = self.get_lower_arc(edge);
        return lower_arc.intersection(&edge.ray, directrix).unwrap().y;
    }

    pub fn get_slot_id_at(&self, slot: &Slot, site: &Point) -> u32 {
        if slot.is_leaf() {
            return slot.id;
        }
        if let Node::Edge(edge) = &slot.value {
            let directrix = site.x;
            if self.get_endpoint_y(edge, directrix) >= site.y {
                return self.get_slot_id_at(&self.nodes[&edge.lower_child], site);
            } else {
                return self.get_slot_id_at(&self.nodes[&edge.upper_child], site);
            }
        }
        panic!("You shouldn't reach here.")
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
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

pub fn fortunes() -> HashMap<usize, Site> {
    let mut rng = rand::rng();

    let mut sites = HashMap::<usize, Site>::new();

    let num_sites = 5;

    let mut events = BinaryHeap::new();

    // // random
    // for idx in 0..num_sites {
    //     sites.insert(
    //         idx,
    //         Site {
    //             location: Point {
    //                 x: rng.random(),
    //                 y: rng.random(),
    //             },
    //         },
    //     );
    // }

    // // in a line at y=0
    // sites.insert(
    //     0,
    //     Site {
    //         location: Point { x: 0., y: 0. },
    //     },
    // );
    // sites.insert(
    //     1,
    //     Site {
    //         location: Point { x: 1., y: 0. },
    //     },
    // );
    // sites.insert(
    //     2,
    //     Site {
    //         location: Point { x: 2., y: 0. },
    //     },
    // );

    // in a right-opening v
    sites.insert(
        0,
        Site {
            location: Point { x: 0., y: 0. },
        },
    );
    sites.insert(
        1,
        Site {
            location: Point { x: 1., y: 1. },
        },
    );
    sites.insert(
        2,
        Site {
            location: Point { x: 1.01, y: -1. },
        },
    );

    // let mut edges = Vec::new();

    for idx in 0..sites.len() {
        events.push(Event::Site(sites[&idx]));
    }

    let mut beachline = Beachline::new();

    // let mut events_vec = Vec::new();
    while let Some(event) = events.pop() {
        match event {
            Event::Site(site) => {
                println!("handling site event");
                let arc_slot_ids = beachline.add_site(&site);
                for arc_slot_id in arc_slot_ids {
                    let circle_event = beachline.check_for_circle_events(arc_slot_id);
                    println!("circle event: {:?}", circle_event);
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
            }
            Event::Circle(circle_event) => {
                println!("handling circle event");
                beachline.remove_arc(circle_event.arc_slot_id);
            }
        }
        // events_vec.push(event);
        // let arc = BeachlineElement::Arc(&event);
        // beachline.push_back(arc);
        println!("beachline: {}", beachline);
        println!("complete edges: {:?}", beachline.complete_edges);
    }
    let mut boundaries = Polyline::new();
    boundaries.points.push(Point { x: -2.0, y: -2.0 });
    boundaries.points.push(Point { x: -2.0, y: 2.0 });
    boundaries.points.push(Point { x: 2.0, y: 2.0 });
    boundaries.points.push(Point { x: 2.0, y: -2.0 });
    boundaries.points.push(Point { x: -2.0, y: -2.0 });

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
                    .set("cx", arc.focus.x)
                    .set("cy", arc.focus.y)
                    .set("r", 0.1);

                document = document.add(path);
            }
            Node::Edge(edge) => {
                println!("handling edge {:?}", edge);
                match boundaries.nearest_intersection(edge.ray) {
                    Some(point) => {
                        println!("intersection point {:?}", point);
                        // edge.length = Some(edge.ray.project(&point));
                        // println!("plotting edge {:?}", edge);
                        let path = get_path_from_points(&edge.ray.start, &point);

                        document = document.add(path);
                    }
                    None => {
                        println!("found no intersection with boundaries")
                    }
                }
            }
        }
    }

    for edge in beachline.complete_edges {
        let path = get_path_from_edge(&edge);

        document = document.add(path);
    }

    svg::save("image.svg", &document).unwrap();

    // println!("{:?}", events_vec);

    return sites;
}

fn get_path_from_edge(edge: &Edge) -> Path {
    let start = edge.ray.start;
    let end = Point {
        x: start.x + edge.ray.direction.x * edge.length.unwrap(),
        y: start.y + edge.ray.direction.y * edge.length.unwrap(),
    };
    return get_path_from_points(&start, &end);
}

fn get_path_from_points(start: &Point, end: &Point) -> Path {
    let data = Data::new()
        .move_to((start.x, start.y))
        .line_by((end.x - start.x, end.y - start.y))
        .close();

    return Path::new()
        .set("fill", "none")
        .set("stroke", "black")
        .set("stroke-width", 0.1)
        .set("d", data);
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_fortunes() {
        assert_eq!(fortunes(), HashMap::new());
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
