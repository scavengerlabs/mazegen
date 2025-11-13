use clipper::clip;
use geometry::{Direction, LineSegment, Parabola, Point, Polyline, Ray};
use rand::Rng;
use std::cmp;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use svg::node::element::path::Data;
use svg::node::element::Circle;
use svg::node::element::Path;
use svg::Document;
use wilsons::wilsons;

#[path = "geometry.rs"]
mod geometry;

#[path = "clipper.rs"]
mod clipper;

#[path = "wilsons.rs"]
mod wilsons;

#[derive(Debug)]
struct Slot {
    id: u32,
    value: Node,
    lower_neighbor: Option<u32>,
    upper_neighbor: Option<u32>,
    parent: Option<u32>,
}

static GLOBAL_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

fn generate_id() -> u32 {
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
                return format!("a{:03} {}", self.id, arc.focus.location);
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

#[derive(Clone, Copy)]
struct SlotBuilder {
    id: u32,
    value: Option<Node>,
    lower_neighbor: Option<u32>,
    upper_neighbor: Option<u32>,
    parent: Option<u32>,
}

impl SlotBuilder {
    pub fn new() -> Self {
        return SlotBuilder {
            id: generate_id(),
            value: None,
            lower_neighbor: None,
            upper_neighbor: None,
            parent: None,
        };
    }

    pub fn build(&self) -> Slot {
        let value = self.value.unwrap();
        return Slot {
            id: self.id,
            value: value,
            lower_neighbor: self.lower_neighbor,
            upper_neighbor: self.upper_neighbor,
            parent: self.parent,
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
    separator_id: u32, // an ID for the boundary separating two adjacent sites - there are often two Edges/Rays per separator
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

    pub fn replace_child(&mut self, old_child_id: u32, new_child_id: u32) {
        if self.lower_child == old_child_id {
            self.lower_child = new_child_id;
        } else if self.upper_child == old_child_id {
            self.upper_child = new_child_id;
        } else {
            panic!("The old child is not here.")
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Arc {
    focus: Site,
}

impl Arc {
    pub fn new(focus: Site) -> Self {
        return Arc { focus: focus };
    }

    pub fn intersection(&self, ray: &Ray, directrix: f32) -> Option<Point> {
        return self.get_parabola(directrix).intersection(ray);
    }

    pub fn get_parabola(&self, directrix: f32) -> Parabola {
        return self.focus.location.to_parabola(directrix);
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
    complete_edges: Vec<(u32, LineSegment)>,
    complete_sites: Vec<Site>,
    cell_separators: HashMap<u32, HashSet<u32>>,
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
            cell_separators: HashMap::new(),
        };
    }

    pub fn sorted_edges(&self, root: u32) -> Vec<(u32, Edge)> {
        match self.nodes[&root].value {
            Node::Edge(edge) => {
                return vec![
                    self.sorted_edges(edge.upper_child),
                    vec![(root, edge)],
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
        let parent_id = old_node.parent;
        match parent_id {
            Some(parent_id) => {
                let parent_edge = self.get_mut_edge(&parent_id);
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

    fn get_lower_neighbor(&self, target_id: u32) -> Option<u32> {
        let target = &self.nodes[&target_id];
        return target.lower_neighbor;
    }

    fn get_upper_neighbor(&self, target_id: u32) -> Option<u32> {
        let target = &self.nodes[&target_id];
        return target.upper_neighbor;
    }

    fn assign_lower_neighbor(&mut self, target_id: u32, lower_neighbor_id: u32) {
        let target = self.nodes.get_mut(&target_id).unwrap();
        target.lower_neighbor = Some(lower_neighbor_id);
    }

    fn assign_upper_neighbor(&mut self, target_id: u32, upper_neighbor_id: u32) {
        let target = self.nodes.get_mut(&target_id).unwrap();
        target.upper_neighbor = Some(upper_neighbor_id);
    }

    fn get_mut_edge(&mut self, slot_id: &u32) -> &mut Edge {
        let slot = self.nodes.get_mut(&slot_id).unwrap();
        return slot.value.get_mut_edge().unwrap();
    }

    fn get_arc(&self, slot_id: &u32) -> Option<Arc> {
        let slot = &self.nodes[&slot_id];
        return slot.value.get_arc();
    }

    fn get_edge(&self, slot_id: &u32) -> Option<Edge> {
        let slot = &self.nodes[&slot_id];
        return slot.value.get_edge();
    }

    fn get_parent_id(&mut self, slot_id: &u32) -> Option<u32> {
        return self.nodes[slot_id].parent;
    }

    fn get_mut_parent_id(&mut self, slot_id: &u32) -> &mut Option<u32> {
        return &mut self.nodes.get_mut(slot_id).unwrap().parent;
    }

    fn set_parent_id(&mut self, slot_id: &u32, parent_id: Option<u32>) {
        *self.get_mut_parent_id(slot_id) = parent_id;
    }

    fn get_site_id(&self, arc_slot_id: &u32) -> u32 {
        return self.nodes[arc_slot_id].value.get_arc().unwrap().focus.id;
    }

    fn get_separator_id(&self, edge_slot_id: &u32) -> u32 {
        return self.nodes[edge_slot_id]
            .value
            .get_edge()
            .unwrap()
            .separator_id;
    }

    // each edge corresponds to a y value where two arcs collide
    pub fn remove_arc(&mut self, target_arc_slot_id: u32, directrix: f32) -> Vec<u32> {
        if !self.nodes.contains_key(&target_arc_slot_id) {
            return vec![];
        }
        // for a circle event, edges should exist on both sides
        let lower_edge_slot_id = self.get_lower_neighbor(target_arc_slot_id).unwrap();
        let upper_edge_slot_id = self.get_upper_neighbor(target_arc_slot_id).unwrap();
        let lower_arc_slot_id = self.get_lower_neighbor(lower_edge_slot_id).unwrap();
        let upper_arc_slot_id = self.get_upper_neighbor(upper_edge_slot_id).unwrap();
        let lower_edge_slot = &self.nodes[&lower_edge_slot_id];
        let upper_edge_slot = &self.nodes[&upper_edge_slot_id];
        let lower_site = self.get_arc(&lower_arc_slot_id).unwrap().focus.location;
        let upper_site = self.get_arc(&upper_arc_slot_id).unwrap().focus.location;
        let lower_edge = &lower_edge_slot.value.get_edge().unwrap();
        let upper_edge = &upper_edge_slot.value.get_edge().unwrap();
        let new_start = lower_edge.ray.intersection(upper_edge.ray).unwrap();
        let mut new_direction = lower_site.perpendicular(&upper_site);
        let normal = lower_site.to_parabola(directrix).normal(&new_start);
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
                let parent_slot_id = self.get_parent_id(&lower_edge_slot_id);
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
                let parent_slot_id = self.get_parent_id(&upper_edge_slot_id);
                (
                    edge_slot_id_to_remove,
                    edge_slot_id_to_replace,
                    new_child_slot_id,
                    parent_slot_id,
                )
            } else {
                panic!("Impossible!");
            };
        if let Some(parent_slot_id) = parent_slot_id {
            let parent_edge = self.get_mut_edge(&parent_slot_id);
            parent_edge.replace_child(edge_slot_id_to_remove, new_child_slot_id);
        }
        self.set_parent_id(&new_child_slot_id, parent_slot_id);

        let edge_to_replace = self.get_edge(&edge_slot_id_to_replace).unwrap();
        let new_edge = Edge {
            ray: new_ray,
            lower_child: edge_to_replace.lower_child,
            upper_child: edge_to_replace.upper_child,
            separator_id: generate_id(),
        };
        let mut new_edge_slot_builder = Slot::builder();
        new_edge_slot_builder.lower_neighbor = Some(lower_arc_slot_id);
        new_edge_slot_builder.upper_neighbor = Some(upper_arc_slot_id);
        new_edge_slot_builder.value = Some(Node::Edge(new_edge));
        new_edge_slot_builder.parent = self.get_parent_id(&edge_slot_id_to_replace);

        self.set_parent_id(&edge_to_replace.lower_child, Some(new_edge_slot_builder.id));
        self.set_parent_id(&edge_to_replace.upper_child, Some(new_edge_slot_builder.id));
        self.replace_slot(edge_slot_id_to_replace, new_edge_slot_builder.id);

        self.add_slot(new_edge_slot_builder);
        // fix neighbors
        let lower_arc_slot_mut = self.nodes.get_mut(&lower_arc_slot_id).unwrap();
        lower_arc_slot_mut.upper_neighbor = Some(new_edge_slot_builder.id);
        let upper_arc_slot_mut = self.nodes.get_mut(&upper_arc_slot_id).unwrap();
        upper_arc_slot_mut.lower_neighbor = Some(new_edge_slot_builder.id);
        let mut arcs_to_check = vec![];
        if let Some(lower_arc_slot_id) = self.nodes[&lower_edge_slot_id].lower_neighbor {
            arcs_to_check.push(lower_arc_slot_id);
        }
        if let Some(upper_arc_slot_id) = self.nodes[&upper_edge_slot_id].upper_neighbor {
            arcs_to_check.push(upper_arc_slot_id);
        }

        // assign lower_edge to target_site and lower_arc
        // assign upper_edge to target_site and upper_arc
        let lower_site_id = self.get_site_id(&lower_arc_slot_id);
        let target_site_id = self.get_site_id(&target_arc_slot_id);
        let upper_site_id = self.get_site_id(&upper_arc_slot_id);
        let lower_separator_id = self.get_separator_id(&lower_edge_slot_id);
        let upper_separator_id = self.get_separator_id(&upper_edge_slot_id);
        self.cell_separators
            .entry(lower_site_id)
            .or_insert(HashSet::new())
            .insert(lower_separator_id);
        let target_cell_separators = self
            .cell_separators
            .entry(target_site_id)
            .or_insert(HashSet::new());
        target_cell_separators.insert(lower_separator_id);
        target_cell_separators.insert(upper_separator_id);
        self.cell_separators
            .entry(upper_site_id)
            .or_insert(HashSet::new())
            .insert(upper_separator_id);

        // wrap up
        let target_arc = self.get_arc(&target_arc_slot_id).unwrap();
        self.complete_sites.push(target_arc.focus);
        // remove obsolete edges
        self.nodes.remove(&lower_edge_slot_id);
        self.nodes.remove(&upper_edge_slot_id);
        // remove obsolete node
        self.nodes.remove(&target_arc_slot_id);

        let (lower_length, upper_length) = lower_edge.ray.terminate(upper_edge.ray).unwrap();
        self.complete_edges.push((
            lower_edge.separator_id,
            lower_edge.to_line_segment(lower_length),
        ));
        self.complete_edges.push((
            upper_edge.separator_id,
            upper_edge.to_line_segment(upper_length),
        ));

        return arcs_to_check;
    }

    fn get_lower_edge(&self, arc_slot_id: u32) -> Option<Edge> {
        let target_arc_slot = &self.nodes[&arc_slot_id];
        match target_arc_slot.lower_neighbor {
            Some(id) => {
                return Some(self.get_edge(&id).unwrap());
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
                let d = intersection.distance_from(
                    &self.nodes[&arc_slot_id]
                        .value
                        .get_arc()
                        .unwrap()
                        .focus
                        .location,
                );
                // the arc should be removed and the edges terminated when the directrix is distance d from the intersection
                return Some(intersection.x + d);
            }
            None => {
                return None;
            }
        }
    }

    fn add_slot(&mut self, slot_builder: SlotBuilder) {
        self.nodes.insert(slot_builder.id, slot_builder.build());
    }

    pub fn add_site(&mut self, site: &Site) -> Vec<u32> {
        if self.root.is_none() {
            let mut new_arc_slot_builder = Slot::builder();
            let new_arc = Arc::new(*site);
            new_arc_slot_builder.value = Some(Node::Arc(new_arc));
            self.root = Some(new_arc_slot_builder.id);
            self.add_slot(new_arc_slot_builder);
            return vec![];
        }
        // get arc and slot that will be replaced with new subtree
        let target_slot_id = self.get_slot_id_at(&self.nodes[&self.root.unwrap()], &site.location);
        let target_slot = &self.nodes[&target_slot_id];
        let target_arc = target_slot.value.get_arc().unwrap();

        // create all slot builders; assigns IDs
        let mut bottom_arc_slot_builder = Slot::builder();
        let mut new_arc_slot_builder = Slot::builder();
        let mut top_arc_slot_builder = Slot::builder();
        let mut bottom_edge_slot_builder = Slot::builder();
        let mut top_edge_slot_builder = Slot::builder();

        // create new arcs
        let bottom_arc = Arc::new(target_arc.focus);
        bottom_arc_slot_builder.value = Some(Node::Arc(bottom_arc));
        bottom_arc_slot_builder.parent = Some(bottom_edge_slot_builder.id);
        let new_arc = Arc::new(*site);
        new_arc_slot_builder.value = Some(Node::Arc(new_arc));
        new_arc_slot_builder.parent = Some(bottom_edge_slot_builder.id);
        let top_arc = Arc::new(target_arc.focus);
        top_arc_slot_builder.value = Some(Node::Arc(top_arc));
        top_arc_slot_builder.parent = Some(top_edge_slot_builder.id);

        // geometry
        let (up_ray, down_ray) = target_arc.split(site);

        // create new edges
        let separator_id = generate_id();
        let bottom_edge = Edge {
            ray: down_ray,
            lower_child: bottom_arc_slot_builder.id,
            upper_child: new_arc_slot_builder.id,
            separator_id: separator_id,
        };
        bottom_edge_slot_builder.value = Some(Node::Edge(bottom_edge));
        bottom_edge_slot_builder.parent = Some(top_edge_slot_builder.id);
        let top_edge = Edge {
            ray: up_ray,
            lower_child: bottom_edge_slot_builder.id,
            upper_child: top_arc_slot_builder.id,
            separator_id: separator_id,
        };
        top_edge_slot_builder.value = Some(Node::Edge(top_edge));
        top_edge_slot_builder.parent = target_slot.parent;

        // assign neighbors
        let target_slot_upper_neighbor_id = target_slot.upper_neighbor;
        let target_slot_lower_neighbor_id = target_slot.lower_neighbor;
        top_arc_slot_builder.upper_neighbor = target_slot_upper_neighbor_id;
        top_arc_slot_builder.lower_neighbor = Some(top_edge_slot_builder.id);
        top_edge_slot_builder.upper_neighbor = Some(top_arc_slot_builder.id);
        top_edge_slot_builder.lower_neighbor = Some(new_arc_slot_builder.id);
        new_arc_slot_builder.upper_neighbor = Some(top_edge_slot_builder.id);
        new_arc_slot_builder.lower_neighbor = Some(bottom_edge_slot_builder.id);
        bottom_edge_slot_builder.upper_neighbor = Some(new_arc_slot_builder.id);
        bottom_edge_slot_builder.lower_neighbor = Some(bottom_arc_slot_builder.id);
        bottom_arc_slot_builder.upper_neighbor = Some(bottom_edge_slot_builder.id);
        bottom_arc_slot_builder.lower_neighbor = target_slot_lower_neighbor_id;

        // fix neighbor links around inserted subtree
        if let Some(id) = target_slot_upper_neighbor_id {
            self.assign_lower_neighbor(id, top_arc_slot_builder.id);
        }
        if let Some(id) = target_slot_lower_neighbor_id {
            self.assign_upper_neighbor(id, bottom_arc_slot_builder.id);
        }

        // attach new subtree
        self.add_slot(top_arc_slot_builder);
        self.add_slot(new_arc_slot_builder);
        self.add_slot(bottom_edge_slot_builder);
        self.add_slot(bottom_arc_slot_builder);
        self.add_slot(top_edge_slot_builder);
        self.replace_slot(target_slot_id, top_edge_slot_builder.id);

        // remove obsolete node
        self.nodes.remove(&target_slot_id);

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
    id: u32,
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
            id: generate_id(),
            location: Point {
                x: point.0,
                y: point.1,
            },
        });
    }

    return sites;
}

fn fortunes(
    sites: Vec<Site>,
    boundaries: &Polyline,
) -> (Beachline, HashSet<(u32, u32)>, HashMap<u32, HashSet<u32>>) {
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
        println!("cell_separators: {:?}", beachline.cell_separators);
        // println!("complete edges: {:?}", beachline.complete_edges);
    }

    // complete rays
    for (edge_slot_id, edge) in beachline.sorted_edges(beachline.root.unwrap()) {
        match boundaries.furthest_intersection(edge.ray) {
            Some(point) => {
                let separator_id = edge.separator_id;
                if let Some(lower_arc_id) = beachline.get_lower_neighbor(edge_slot_id) {
                    let lower_site_id = beachline.nodes[&lower_arc_id]
                        .value
                        .get_arc()
                        .unwrap()
                        .focus
                        .id;
                    beachline
                        .cell_separators
                        .entry(lower_site_id)
                        .or_insert(HashSet::new())
                        .insert(separator_id);
                }
                if let Some(upper_arc_id) = beachline.get_upper_neighbor(edge_slot_id) {
                    let upper_site_id = beachline.nodes[&upper_arc_id]
                        .value
                        .get_arc()
                        .unwrap()
                        .focus
                        .id;
                    beachline
                        .cell_separators
                        .entry(upper_site_id)
                        .or_insert(HashSet::new())
                        .insert(separator_id);
                }
                beachline.complete_edges.push((
                    edge.separator_id,
                    LineSegment {
                        first: edge.ray.start,
                        second: point,
                    },
                ));
            }
            None => {
                println!("found no intersection with boundaries")
            }
        }
    }

    let mut edges_by_separator = HashMap::new();
    for (separator_id, segment) in &beachline.complete_edges {
        edges_by_separator
            .entry(separator_id)
            .or_insert(vec![])
            .push(segment);
    }
    let mut site_by_id = HashMap::new();
    for site in &beachline.complete_sites {
        site_by_id.insert(site.id, *site);
    }
    for (_, node) in &beachline.nodes {
        if let Node::Arc(arc) = node.value {
            site_by_id.insert(arc.focus.id, arc.focus);
        }
    }
    let mut separators_by_site = HashMap::new();
    for (site_id, separator_ids) in &beachline.cell_separators {
        let site = site_by_id[site_id];
        let mut lines = vec![];
        for separator_id in separator_ids {
            // get line from first segment (they're colinear)
            lines.push((*separator_id, edges_by_separator[separator_id][0].to_line()));
        }
        let clipped = clip(boundaries, &lines, &site.location);
        for (_, separators) in &clipped {
            for separator in separators {
                separators_by_site
                    .entry(*site_id)
                    .or_insert(HashSet::new())
                    .insert(separator.clone());
            }
        }
    }

    let mut sites_by_separator = HashMap::new();
    for (site, separators) in &separators_by_site {
        for separator in separators {
            sites_by_separator
                .entry(*separator)
                .or_insert(HashSet::new())
                .insert(*site);
        }
    }
    let mut graph = HashMap::new();
    for sites in sites_by_separator.values() {
        assert_eq!(sites.len(), 2);
        let sites_vec: Vec<&u32> = sites.into_iter().collect();
        graph
            .entry(*sites_vec[0])
            .or_insert(HashSet::new())
            .insert(sites_vec[1]);
        graph
            .entry(*sites_vec[1])
            .or_insert(HashSet::new())
            .insert(sites_vec[0]);
    }
    let connections = wilsons(&graph);

    return (beachline, connections, sites_by_separator);
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

fn plot(
    beachline: &Beachline,
    boundaries: &Polyline,
    sites_by_separator: HashMap<u32, HashSet<u32>>,
    connections: HashSet<(u32, u32)>,
    width: f32,
) {
    let mut document = Document::new().set("viewBox", (-2.5, -2.5, 5, 5));

    for idx in 0..boundaries.points.len() {
        let start = boundaries.points[idx];
        let end = boundaries.points[(idx + 1) % boundaries.points.len()];
        let data = Data::new()
            .move_to((start.x, start.y))
            .line_by((end.x - start.x, end.y - start.y))
            .close();

        let path = Path::new()
            .set("fill", "none")
            .set("opacity", 0.3)
            .set("stroke", "purple")
            .set("stroke-width", width)
            .set("d", data);

        document = document.add(path);
    }

    let mut edges_by_separator = HashMap::new();
    for (separator_id, segment) in &beachline.complete_edges {
        edges_by_separator
            .entry(separator_id)
            .or_insert(vec![])
            .push(segment);
    }
    let mut site_by_id = HashMap::new();
    for site in &beachline.complete_sites {
        site_by_id.insert(site.id, *site);
    }
    for (_, node) in &beachline.nodes {
        if let Node::Arc(arc) = node.value {
            site_by_id.insert(arc.focus.id, arc.focus);
        }
    }

    for site in site_by_id.values() {
        let path = Circle::new()
            .set("fill", "black")
            .set("opacity", 0.3)
            .set("cx", site.location.x)
            .set("cy", site.location.y)
            .set("r", width);

        document = document.add(path);
    }

    // let mut separators_by_site = HashMap::new();
    for (site_id, separator_ids) in &beachline.cell_separators {
        let site = site_by_id[site_id];
        let mut lines = vec![];
        for separator_id in separator_ids {
            // get line from first segment (they're colinear)
            lines.push((*separator_id, edges_by_separator[separator_id][0].to_line()));
        }
        let clipped = clip(boundaries, &lines, &site.location);
        println!("clipped: {:?}", clipped);

        for idx in 0..clipped.len() {
            let (start, start_separators) = &clipped[idx];
            let (end, end_separators) = &clipped[(idx + 1) % clipped.len()];
            let both_separators: Vec<u32> = start_separators
                .intersection(end_separators)
                .copied()
                .collect();
            if !both_separators.is_empty() {
                assert_eq!(both_separators.len(), 1);
                let separator_id = both_separators[0];
                let sites: Vec<u32> = sites_by_separator[&separator_id].iter().copied().collect();
                if connections.contains(&(sites[0], sites[1]))
                    || connections.contains(&(sites[1], sites[0]))
                {
                    // don't draw this separator
                    continue;
                }
            }
            let data = Data::new()
                .move_to((start.x, start.y))
                .line_by((end.x - start.x, end.y - start.y))
                .close();

            let path = Path::new()
                .set("fill", "none")
                .set("stroke", "purple")
                .set("opacity", 0.3)
                .set("stroke-width", width)
                .set("d", data);

            document = document.add(path);
        }
        let mut clipped_points = vec![];
        for (point, _) in &clipped {
            clipped_points.push(*point);
        }

        let centroid = Polyline {
            points: clipped_points,
        }
        .centroid();
        let path = Circle::new()
            .set("fill", "orange")
            .set("opacity", 0.3)
            .set("cx", centroid.x)
            .set("cy", centroid.y)
            .set("r", width);

        document = document.add(path);
        // break;
    }

    for (first, second) in connections {
        let start = site_by_id[&first].location;
        let end = site_by_id[&second].location;
        let data = Data::new()
            .move_to((start.x, start.y))
            .line_by((end.x - start.x, end.y - start.y))
            .close();

        let path = Path::new()
            .set("fill", "none")
            .set("opacity", 0.3)
            .set("stroke", "gray")
            .set("stroke-width", width)
            .set("d", data);

        document = document.add(path);
    }

    svg::save("image.svg", &document).unwrap();
}

fn run_fortunes(
    points: Vec<(f32, f32)>,
    boundary_points: Vec<(f32, f32)>,
) -> Vec<(u32, LineSegment)> {
    let sites = add_sites(points);

    let mut boundaries = Polyline::new();
    for point in boundary_points {
        boundaries.points.push(Point {
            x: point.0,
            y: point.1,
        });
    }

    let (beachline, connections, sites_by_separator) = fortunes(sites, &boundaries);

    plot(
        &beachline,
        &boundaries,
        sites_by_separator,
        connections,
        0.05,
    );

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

    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];

    run_fortunes(sites, boundary_polyline);
}
