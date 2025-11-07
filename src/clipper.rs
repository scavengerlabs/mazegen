use super::geometry::{Direction, Line, LineSegment, Point, Polyline, Ray};

pub fn clip(subject: Polyline, clip_edges: Vec<Line>, site: Point) -> Vec<Point> {
    // subject is closed (end connects to beginning)
    // assume that clip is convex
    // clip may not be closed
    // site is inside the clip (semi-)polygon

    let mut outputs = subject.points;

    for line in clip_edges {
        let inputs = outputs.clone();
        outputs.clear();
        let normal_direction = Direction {
            x: line.direction.y,
            y: -line.direction.x,
        };
        let mut normal = Ray {
            start: line.start,
            direction: normal_direction,
        };
        if normal.project(&site) < 0.0 {
            normal = -normal;
        }
        for idx in 0..inputs.len() {
            let next = inputs[(idx + 1) % inputs.len()];
            let current = inputs[idx];
            let segment = LineSegment {
                first: next,
                second: current,
            };
            match segment.intersection_with_line(line) {
                Some(intersection) => {
                    outputs.push(intersection);
                }
                None => {}
            }
            if normal.project(&next) >= 0.0 {
                outputs.push(next);
            }
        }
    }
    return outputs;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip() {
        let subject = Polyline {
            points: vec![
                Point { x: -1.0, y: -1.0 },
                Point { x: 1.0, y: -1.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: -1.0, y: 1.0 },
            ],
        };
        let clip_edges = vec![Line {
            start: Point { x: 0.0, y: 0.0 },
            direction: Direction::new(1.0, 0.0),
        }];
        let site = Point { x: 0.0, y: -1.0 };

        let points = clip(subject, clip_edges, site);

        let expected_points = vec![
            Point { x: 1.0, y: -1.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: -1.0, y: 0.0 },
            Point { x: -1.0, y: -1.0 },
        ];
        println!("points: {:?}", points);
        for (point, expected_point) in points.iter().zip(expected_points.iter()) {
            assert!(point.close_to(expected_point, 0.001));
        }
    }
}
