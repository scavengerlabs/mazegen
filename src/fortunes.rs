use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::iter::Map;
use std::ops::Neg;
use std::ptr;

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

// impl Ord for Point {
//     fn cmp(&self, other: &Self) -> Ordering {
//         // Compare based on x only
//         self.x.cmp(&other.x)
//     }
// }

// impl PartialOrd for Point {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        return Point {
            x: x, // NotNan::new(x).expect("nan"),
            y: y, // NotNan::new(y).expect("nan"),
        };
    }

    pub fn distance_from(&self, other: &Point) -> f32 {
        // compute Euclidean distance between points
        return ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).sqrt();
    }
}

#[derive(Copy, Clone, Debug)]
struct Direction {
    x: f32,
    y: f32,
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
        return Some(valid_options[0].clone());
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
    value: Node,
}

pub enum LowerNeighborOption<T> {
    Below(T),
    Above,
    None,
}

impl Slot {
    pub fn from_arc(arc: Arc) -> Self {
        return Self {
            value: Node::Arc(arc),
        };
    }

    pub fn from_edge(edge: Edge) -> Self {
        return Self {
            value: Node::Edge(edge),
        };
    }

    pub fn is_leaf(&self) -> bool {
        match self.value {
            Node::Edge(_) => true,
            _ => false,
        }
    }

    pub fn get_slot_at(&mut self, site: &Point) -> &mut Slot {
        if self.is_leaf() {
            return self;
        }
        if let Node::Edge(edge) = &mut self.value {
            let directrix = site.x;
            if edge.get_endpoint_y(directrix) < site.y {
                return edge.lower_child.get_slot_at(site);
            } else {
                return edge.upper_child.get_slot_at(site);
            }
        }
        panic!("You shouldn't reach here.")
    }

    // pub fn get_nearest_edge_below<'a, 'b>(
    //     &'a mut self,
    //     arc: &Arc,
    // ) -> LowerNeighborOption<&'b mut Slot>
    // where
    //     'a: 'b,
    // {
    //     if self.is_leaf() {
    //         return LowerNeighborOption::None;
    //     }
    //     if let Node::Edge(edge) = &mut self.value {
    //         // if edge.upper_child is site
    //         //   return edge
    //         if let Node::Arc(upper_arc) = &edge.upper_child.value {
    //             if ptr::eq(upper_arc, ptr::from_ref(arc)) {
    //                 return LowerNeighborOption::Below(self);
    //             }
    //         }
    //         // if edge.lower_child is site
    //         //   return "above"
    //         if let Node::Arc(lower_arc) = &edge.lower_child.value {
    //             if ptr::eq(lower_arc, ptr::from_ref(arc)) {
    //                 return LowerNeighborOption::Above;
    //             }
    //         }
    //         // upper = edge.upper_child.get_nearest_edge_below
    //         // lower = edge.lower_child.get_nearest_edge_below
    //         let upper = edge.upper_child.get_nearest_edge_below(arc);
    //         let lower = edge.lower_child.get_nearest_edge_below(arc);
    //         // if either is an arc, return it
    //         if let LowerNeighborOption::Below(_) = &upper {
    //             return upper;
    //         }
    //         if let LowerNeighborOption::Below(_) = &lower {
    //             return lower;
    //         }
    //         if matches!(lower, LowerNeighborOption::Above) {
    //             // if lower is "above", return "none"
    //             return LowerNeighborOption::None;
    //         } else if matches!(upper, LowerNeighborOption::Above) {
    //             // if upper is "above" and lower is "none", return edge
    //             return LowerNeighborOption::Below(self);
    //         }
    //         return LowerNeighborOption::None;
    //     }
    //     panic!("You shouldn't reach here.")
    // }
}

#[derive(Debug)]
enum Node {
    Arc(Arc),
    Edge(Edge),
}

impl Node {}

#[derive(Debug)]
struct Edge {
    ray: Ray,
    lower_child: Box<Slot>,
    upper_child: Box<Slot>,
}

impl Edge {
    pub fn get_lower_arc(&self) -> &Arc {
        match &self.lower_child.value {
            Node::Edge(edge) => &edge.get_lower_arc(),
            Node::Arc(arc) => arc,
        }
    }
    pub fn get_upper_arc(&self) -> &Arc {
        match &self.upper_child.value {
            Node::Edge(edge) => &edge.get_upper_arc(),
            Node::Arc(arc) => arc,
        }
    }

    pub fn get_endpoint_y(&self, directrix: f32) -> f32 {
        let lower_arc = self.get_lower_arc();
        return lower_arc.intersection(&self.ray, directrix).unwrap().y;
    }
}

#[derive(Debug)]
struct Arc {
    focus: Point,
}

impl Arc {
    pub fn new(focus: Point) -> Self {
        return Arc { focus: focus };
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
    root: Option<Box<Slot>>,
}

impl Beachline {
    pub fn new() -> Self {
        return Beachline { root: None };
    }

    pub fn get_slot_at(&mut self, site: &Point) -> &mut Slot {
        match &mut self.root {
            Some(slot) => {
                return slot.get_slot_at(site);
            }
            None => {
                panic!("Don't call get_arc_at() of an empty Beachline.")
            }
        }
    }

    // pub fn get_lower_edge(&mut self, site: &Site) -> &mut Slot {}

    // each edge corresponds to a y value where two arcs collide

    pub fn add_site(&mut self, site: &Site) {
        let target_slot = self.get_slot_at(&site.location);
        let target_arc = match &target_slot.value {
            Node::Arc(arc) => arc,
            Node::Edge(_) => panic!("don't"),
        };
        let bottom_arc = Arc::new(target_arc.focus);
        let new_arc = Arc::new(site.location);
        let top_arc = Arc::new(target_arc.focus);
        let site_to_arc_ray_start = target_arc
            .intersection(
                &Ray {
                    start: site.location.clone(),
                    direction: Direction::new(-1.0, 0.0),
                },
                site.location.x,
            )
            .unwrap();
        let parabola = target_arc.get_parabola(site.location.x);
        let up_tangent = parabola.tangent_at(&site_to_arc_ray_start);
        let down_tangent = -(up_tangent.clone());
        let down_ray = Ray {
            start: site_to_arc_ray_start.clone(),
            direction: down_tangent,
        };
        let up_ray = Ray {
            start: site_to_arc_ray_start,
            direction: up_tangent,
        };
        let bottom_edge = Edge {
            ray: down_ray.clone(),
            lower_child: Box::new(Slot::from_arc(new_arc)),
            upper_child: Box::new(Slot::from_arc(bottom_arc)),
        };
        let top_edge = Edge {
            ray: up_ray,
            lower_child: Box::new(Slot::from_arc(top_arc)),
            upper_child: Box::new(Slot::from_edge(bottom_edge)),
        };
        target_slot.value = Node::Edge(top_edge);
    }
}

#[derive(Debug)]
pub struct Site {
    location: Point,
}

impl PartialEq for Site {
    fn eq(&self, other: &Self) -> bool {
        self.location.x == other.location.x
    }
}

impl Eq for Site {}

impl Ord for Site {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare based on x only
        self.location.x.total_cmp(&other.location.x)
    }
}

impl PartialOrd for Site {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn fortunes() -> HashMap<i32, Site> {
    let mut rng = rand::thread_rng();

    let mut sites = HashMap::new();

    let num_sites = 10;

    let mut events = BinaryHeap::new();
    for idx in 0..num_sites {
        sites.insert(
            idx,
            Site {
                location: Point {
                    x: rng.gen(),
                    y: rng.gen(),
                },
            },
        );
    }

    // let mut edges = Vec::new();

    for idx in 0..num_sites {
        events.push(sites.get(&idx).unwrap());
    }

    // directrix is an x position, starting at the leftmost site
    let mut directrix = 0.0;

    let mut beachline = Beachline::new();

    let mut events_vec = Vec::new();
    while let Some(event) = events.pop() {
        directrix = event.location.x;
        beachline.add_site(event);
        events_vec.push(event);
        // let arc = BeachlineElement::Arc(&event);
        // beachline.push_back(arc);
    }

    println!("{:?}", beachline);

    // println!("{:?}", events_vec);

    return sites;
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
