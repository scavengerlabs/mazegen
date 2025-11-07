use geometry::{Direction, LineSegment, Parabola, Point, Polyline, Ray};
use rand::Rng;
use std::cmp;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use svg::node::element::path::Data;
use svg::node::element::Circle;
use svg::node::element::Path;
use svg::Document;

#[path = "geometry.rs"]
mod geometry;

#[path = "clipper.rs"]
mod clipper;

#[derive(Debug)]
struct Slot {
    id: u32,
    value: Node,
    lower_neighbor: Option<u32>,
    upper_neighbor: Option<u32>,
}

static GLOBAL_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

fn generate_id() -> u32 {
    GLOBAL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Slot {
    pub fn get_parent_id(&self) -> Option<u32> {
        return match self.value {
            Node::Arc(arc) => arc.parent,
            Node::Edge(edge) => edge.parent,
        };
    }
    pub fn get_mut_parent_id(&mut self) -> &mut Option<u32> {
        return match &mut self.value {
            Node::Arc(ref mut arc) => &mut arc.parent,
            Node::Edge(ref mut edge) => &mut edge.parent,
        };
    }

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

    pub fn get_mut_edge(&mut self) -> Option<&mut Edge> {
        return match self {
            Node::Edge(ref mut edge) => Some(edge),
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
}

impl Edge {
    pub fn to_line_segment(&self, length: f32) -> LineSegment {
        let first = self.ray.start;
        let second = Point {
            x: self.ray.start.x + self.ray.direction.x * length,
            y: self.ray.start.y + self.ray.direction.y * length,
        };
        return LineSegment {
            first: first,
            second: second,
        };
    }
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
        return Parabola {
            focus: self.focus,
            directrix: directrix,
        };
    }

    pub fn split(&self, site: &Site) -> (Ray, Ray) {
        let ray = Ray {
            start: site.location,
            direction: Direction::new(-1.0, 0.0),
        };
        let site_to_arc_ray_start = self.intersection(&ray, site.location.x).unwrap();
        let parabola = self.get_parabola(site.location.x);
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
        return (up_ray, down_ray);
    }
}

#[derive(Debug)]
struct Beachline {
    root: Option<u32>,
    nodes: HashMap<u32, Slot>,
    complete_edges: Vec<LineSegment>,
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

    pub fn sorted_edges(&self, root: u32) -> Vec<Edge> {
        match self.nodes[&root].value {
            Node::Edge(edge) => {
                return vec![
                    self.sorted_edges(edge.upper_child),
                    vec![edge],
                    self.sorted_edges(edge.lower_child),
                ]
                .concat();
            }
            Node::Arc(_) => {
                return vec![];
            }
        }
    }

    fn replace_slot(&mut self, old_id: u32, new_id: u32) {
        let old_node = &self.nodes[&old_id];
        let parent_id = old_node.get_parent_id();
        match parent_id {
            Some(parent_id) => {
                let parent_slot = self.nodes.get_mut(&parent_id).unwrap();
                let parent_edge = parent_slot.value.get_mut_edge().unwrap();
                if parent_edge.lower_child == old_id {
                    parent_edge.lower_child = new_id;
                } else if parent_edge.upper_child == old_id {
                    parent_edge.upper_child = new_id;
                } else {
                    panic!("This should not be.")
                }
            }
            None => self.root = Some(new_id),
        }
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
        let (edge_slot_id_to_remove, edge_slot_id_to_replace, new_child_slot_id, parent_slot_id) =
            if lower_edge.upper_child == target_arc_slot_id {
                // connect lower_edge.lower_child to lower_edge.parent
                // this cuts out lower_edge and its upper_child (AKA target_arc)
                let edge_slot_id_to_remove = lower_edge_slot_id;
                let edge_slot_id_to_replace = upper_edge_slot_id;
                let new_child_slot_id = lower_edge.lower_child;
                let parent_slot_id = lower_edge.parent;
                (
                    edge_slot_id_to_remove,
                    edge_slot_id_to_replace,
                    new_child_slot_id,
                    parent_slot_id,
                )
            } else if upper_edge.lower_child == target_arc_slot_id {
                // connect upper_edge.upper_child to upper_edge.parent
                // this cuts out upper_edge and its lower_child (AKA target_arc)
                let edge_slot_id_to_remove = upper_edge_slot_id;
                let edge_slot_id_to_replace = lower_edge_slot_id;
                let new_child_slot_id = upper_edge.upper_child;
                let parent_slot_id = upper_edge.parent;
                (
                    edge_slot_id_to_remove,
                    edge_slot_id_to_replace,
                    new_child_slot_id,
                    parent_slot_id,
                )
            } else {
                panic!("Impossible!");
            };
        match parent_slot_id {
            Some(parent_slot_id) => {
                let parent_slot = self.nodes.get_mut(&parent_slot_id).unwrap();
                let parent_edge = parent_slot.value.get_mut_edge().unwrap();
                if parent_edge.lower_child == edge_slot_id_to_remove {
                    parent_edge.lower_child = new_child_slot_id;
                } else if parent_edge.upper_child == edge_slot_id_to_remove {
                    parent_edge.upper_child = new_child_slot_id;
                }
            }
            None => {}
        }
        let new_child_slot = self.nodes.get_mut(&new_child_slot_id).unwrap();
        match new_child_slot.value {
            Node::Arc(ref mut new_child_mut) => {
                new_child_mut.parent = parent_slot_id;
            }
            Node::Edge(ref mut new_child_mut) => {
                new_child_mut.parent = parent_slot_id;
            }
        }
        // fetch upper_edge anew because we've modified self.nodes
        let edge_slot_to_be_replaced = &self.nodes[&edge_slot_id_to_replace];
        let edge_to_be_replaced = &edge_slot_to_be_replaced.value.get_edge().unwrap();
        // create a new_edge between arcs lower_neighbor.lower_neighbor and upper_neighbor.upper_neighbor
        let new_edge = Edge {
            ray: new_ray,
            lower_child: edge_to_be_replaced.lower_child,
            upper_child: edge_to_be_replaced.upper_child,
            parent: edge_to_be_replaced.parent,
        };
        // replace upper_edge with new_edge
        let mut new_edge_slot_builder = Slot::builder();
        new_edge_slot_builder.lower_neighbor = Some(lower_arc_slot_id);
        new_edge_slot_builder.upper_neighbor = Some(upper_arc_slot_id);
        new_edge_slot_builder.value = Some(Node::Edge(new_edge));

        *self
            .nodes
            .get_mut(&edge_to_be_replaced.lower_child)
            .unwrap()
            .get_mut_parent_id() = Some(new_edge_slot_builder.id);

        *self
            .nodes
            .get_mut(&edge_to_be_replaced.upper_child)
            .unwrap()
            .get_mut_parent_id() = Some(new_edge_slot_builder.id);

        self.replace_slot(edge_slot_id_to_replace, new_edge_slot_builder.id);
        self.nodes
            .insert(new_edge_slot_builder.id, new_edge_slot_builder.build());
        // fix neighbors
        let lower_arc_slot_mut = self.nodes.get_mut(&lower_arc_slot_id).unwrap();
        lower_arc_slot_mut.upper_neighbor = Some(new_edge_slot_builder.id);
        let upper_arc_slot_mut = self.nodes.get_mut(&upper_arc_slot_id).unwrap();
        upper_arc_slot_mut.lower_neighbor = Some(new_edge_slot_builder.id);
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

        let (lower_length, upper_length) = lower_edge.ray.terminate(upper_edge.ray).unwrap();
        self.complete_edges
            .push(lower_edge.to_line_segment(lower_length));
        self.complete_edges
            .push(upper_edge.to_line_segment(upper_length));
        // let target_cell_edges = self.cell_edges.get_mut(target_site_id);
        // target_cell_edges.push(lower_edge_id);
        // target_cell_edges.push(upper_edge_id);
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
        let target_slot = &self.nodes[&target_slot_id];
        let target_arc = target_slot.value.get_arc().unwrap();

        // geometry
        let (up_ray, down_ray) = target_arc.split(site);

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
        };
        bottom_edge_slot_builder.value = Some(Node::Edge(bottom_edge));
        let top_edge = Edge {
            ray: up_ray,
            lower_child: bottom_edge_slot_builder.id,
            upper_child: top_arc_slot_builder.id,
            parent: target_arc.parent,
        };

        let target_slot_upper_neighbor_id = target_slot.upper_neighbor;
        let target_slot_lower_neighbor_id = target_slot.lower_neighbor;
        // assign neighbors
        top_arc_slot_builder.upper_neighbor = target_slot_upper_neighbor_id;
        top_arc_slot_builder.lower_neighbor = Some(top_edge_slot_builder.id);
        // target slot becomes "top edge"
        top_edge_slot_builder.upper_neighbor = Some(top_arc_slot_builder.id);
        top_edge_slot_builder.lower_neighbor = Some(new_arc_slot_builder.id);
        new_arc_slot_builder.upper_neighbor = Some(top_edge_slot_builder.id);
        new_arc_slot_builder.lower_neighbor = Some(bottom_edge_slot_builder.id);
        bottom_edge_slot_builder.upper_neighbor = Some(new_arc_slot_builder.id);
        bottom_edge_slot_builder.lower_neighbor = Some(bottom_arc_slot_builder.id);
        bottom_arc_slot_builder.upper_neighbor = Some(bottom_edge_slot_builder.id);
        bottom_arc_slot_builder.lower_neighbor = target_slot_lower_neighbor_id;

        // attach new subtree
        top_edge_slot_builder.value = Some(Node::Edge(top_edge));
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
        self.nodes
            .insert(top_edge_slot_builder.id, top_edge_slot_builder.build());
        self.replace_slot(target_slot_id, top_edge_slot_builder.id);

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

fn add_sites(points: Vec<(f32, f32)>) -> Vec<Site> {
    let mut sites = vec![];

    for point in points {
        sites.push(Site {
            location: Point {
                x: point.0,
                y: point.1,
            },
        });
    }

    return sites;
}

fn fortunes(sites: Vec<Site>, boundaries: &Polyline) -> Beachline {
    let mut events = BinaryHeap::new();

    for site in sites {
        events.push(Event::Site(site));
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
    for edge in beachline.sorted_edges(beachline.root.unwrap()) {
        match boundaries.furthest_intersection(edge.ray) {
            Some(point) => {
                beachline.complete_edges.push(LineSegment {
                    first: edge.ray.start,
                    second: point,
                });
            }
            None => {
                println!("found no intersection with boundaries")
            }
        }
    }

    return beachline;
}

fn get_path_from_points(start: &Point, end: &Point, color: &str, width: f32) -> Path {
    let data = Data::new()
        .move_to((start.x, start.y))
        .line_by((end.x - start.x, end.y - start.y))
        .close();

    return Path::new()
        .set("fill", "none")
        .set("stroke", color)
        .set("stroke-width", width)
        .set("opacity", 0.3)
        .set("d", data);
}

fn plot(beachline: &Beachline, boundaries: &Polyline, width: f32) {
    let mut document = Document::new().set("viewBox", (-2.5, -2.5, 5, 5));

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
            .set("stroke-width", width)
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
                    .set("r", width);

                document = document.add(path);
            }
            Node::Edge(_) => {}
        }
    }

    for segment in &beachline.complete_edges {
        let path = get_path_from_points(&segment.first, &segment.second, "green", width);
        document = document.add(path);
    }

    for point in &beachline.complete_sites {
        let path = Circle::new()
            .set("fill", "blue")
            .set("opacity", 0.3)
            .set("cx", point.x)
            .set("cy", point.y)
            .set("r", width);

        document = document.add(path);
    }

    svg::save("image.svg", &document).unwrap();
}

fn run_fortunes(points: Vec<(f32, f32)>, boundary_points: Vec<(f32, f32)>) -> Vec<LineSegment> {
    let sites = add_sites(points);

    let mut boundaries = Polyline::new();
    for point in boundary_points {
        boundaries.points.push(Point {
            x: point.0,
            y: point.1,
        });
    }

    let beachline = fortunes(sites, &boundaries);

    plot(&beachline, &boundaries, 0.05);

    return beachline.complete_edges;
}

#[cfg(test)]
#[path = "fortunes_tests.rs"]
mod tests;

fn main() {
    let mut rng = rand::rng();
    let mut sites = vec![];
    let num_sites = 100;

    for _ in 0..num_sites {
        sites.push((rng.random::<f32>() * 4. - 2., rng.random::<f32>() * 4. - 2.));
    }

    let boundary_polyline = vec![
        (-2.0, -2.0),
        (-2.0, 2.0),
        (2.0, 2.0),
        (2.0, -2.0),
        (-2.0, -2.0),
    ];

    run_fortunes(sites, boundary_polyline);
}
