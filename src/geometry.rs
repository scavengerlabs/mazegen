use std::fmt;
use std::ops::Neg;

#[derive(Copy, Clone, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
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

    pub fn close_to(&self, other: &Point, epsilon: f32) -> bool {
        let distance = self.distance_from(other);
        return distance <= epsilon;
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

    pub fn to_parabola(self, directrix: f32) -> Parabola {
        return Parabola {
            focus: self,
            directrix: directrix,
        };
    }
}

#[derive(Debug)]
pub struct LineSegment {
    pub first: Point,
    pub second: Point,
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

    pub fn intersection_with_line(&self, line: Line) -> Option<Point> {
        let segment_ray = Ray {
            start: self.first,
            direction: Direction::new(self.second.x - self.first.x, self.second.y - self.first.y),
        };
        match line.intersection(segment_ray) {
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

pub struct Polyline {
    pub points: Vec<Point>,
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
pub struct Direction {
    pub x: f32,
    pub y: f32,
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
pub struct Line {
    pub start: Point,
    pub direction: Direction,
}

impl Line {
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
        if t_2 < 0.0 {
            return None;
        }
        return Some(Point {
            x: self.start.x + self.direction.x * t_1,
            y: self.start.y + self.direction.y * t_1,
        });
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub start: Point,
    pub direction: Direction,
}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}->", self.start, self.direction)
    }
}

impl Neg for Ray {
    type Output = Self;

    fn neg(self) -> Self::Output {
        return Ray {
            start: self.start,
            direction: -self.direction,
        };
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
pub struct Parabola {
    pub focus: Point,
    pub directrix: f32,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
