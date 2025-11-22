use super::geometry::{Direction, Line, LineSegment, Point, Polyline, Ray};
use std::collections::HashSet;

pub fn clip(
    subject: &Polyline,
    clip_edges: &Vec<(u32, Line)>,
    site: &Point,
) -> Vec<(Point, HashSet<u32>)> {
    // subject is closed (end connects to beginning)
    // assume that clip is convex
    // clip may not be closed
    // site is inside the clip (semi-)polygon

    let mut outputs: Vec<(Point, HashSet<u32>)> = vec![];
    for point in &subject.points {
        outputs.push((*point, HashSet::new()));
    }

    for (separator_id, line) in clip_edges {
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
            let (next, next_separator_ids) = &inputs[(idx + 1) % inputs.len()];
            let (current, current_separator_ids) = &inputs[idx];
            let segment = LineSegment {
                first: *next,
                second: *current,
            };
            match segment.intersection_with_line(line) {
                Some(intersection) => {
                    let mut separator_ids = HashSet::new();
                    for id in next_separator_ids.intersection(&current_separator_ids) {
                        separator_ids.insert(*id);
                    }
                    separator_ids.insert(*separator_id);
                    outputs.push((intersection, separator_ids));
                    if outputs.len() >= 2
                        && outputs[outputs.len() - 1].1.len() == 2
                        && outputs[outputs.len() - 1].1 == outputs[outputs.len() - 2].1
                    {
                        // The last two points are identical - they have the same two separators.
                        // Remove both.
                        outputs.pop();
                        outputs.pop();
                    }
                }
                None => {}
            }
            if normal.project(&next) >= 0.0 {
                outputs.push((*next, next_separator_ids.clone()));
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
        let clip_edges = vec![(
            0,
            Line {
                start: Point { x: 0.0, y: 0.0 },
                direction: Direction::new(1.0, 0.0),
            },
        )];
        let site = Point { x: 0.0, y: -1.0 };

        let points = clip(&subject, &clip_edges, &site);

        let expected_points = vec![
            Point { x: 1.0, y: -1.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: -1.0, y: 0.0 },
            Point { x: -1.0, y: -1.0 },
        ];
        println!("points: {:?}", points);
        for ((point, _), expected_point) in points.iter().zip(expected_points.iter()) {
            assert!(point.close_to(expected_point, 0.001));
        }
    }

    #[test]
    fn test_clip_corner() {
        let subject = Polyline {
            points: vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 0.0, y: 2.0 },
                Point { x: 1.0, y: 2.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 2.0, y: 1.0 },
                Point { x: 2.0, y: 0.0 },
            ],
        };
        let clip_edges = vec![(
            0,
            Line {
                start: Point { x: 0.75, y: 1.75 },
                direction: Direction::new(1.0, -1.0),
            },
        )];
        let site = Point { x: 0.0, y: 0.0 };

        let points = clip(&subject, &clip_edges, &site);

        let expected_points = vec![
            Point { x: 0.0, y: 2.0 },
            Point { x: 0.5, y: 2.0 },
            Point { x: 1.0, y: 1.5 },
            Point { x: 1.0, y: 1.0 },
            Point {
                x: 1.5,
                y: 0.99999994,
            },
            Point { x: 2.0, y: 0.5 },
            Point { x: 2.0, y: 0.0 },
            Point { x: 0.0, y: 0.0 },
        ];
        for ((point, _), expected_point) in points.iter().zip(expected_points.iter()) {
            assert!(point.close_to(expected_point, 0.001));
        }
    }

    #[test]
    fn test_clip_split() {
        let subject = Polyline {
            points: vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 0.0, y: 2.0 },
                Point { x: 1.0, y: 2.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 2.0, y: 1.0 },
                Point { x: 2.0, y: 0.0 },
            ],
        };
        let clip_edges = vec![
            (
                0,
                Line {
                    start: Point { x: 0.75, y: 1.75 },
                    direction: Direction::new(1.0, -1.0),
                },
            ),
            (
                1,
                Line {
                    start: Point { x: 1.5, y: 2.5 },
                    direction: Direction::new(1.0, -1.0),
                },
            ),
        ];
        let site = Point { x: 1.0, y: 2.0 };

        let points = clip(&subject, &clip_edges, &site);

        let expected_points = vec![
            Point { x: 1.0, y: 2.0 },
            Point { x: 1.0, y: 1.5 },
            Point {
                x: 1.5,
                y: 0.99999994,
            },
            Point { x: 2.0, y: 1.0 },
            Point { x: 2.0, y: 0.5 },
            Point { x: 0.5, y: 2.0 },
        ];
        for ((point, _), expected_point) in points.iter().zip(expected_points.iter()) {
            assert!(point.close_to(expected_point, 0.001));
        }
    }

    #[test]
    fn test_clip_dangler() {
        let subject = Polyline {
            points: vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 0.0, y: 2.0 },
                Point { x: 1.0, y: 2.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 2.0, y: 1.0 },
                Point { x: 2.0, y: 0.0 },
            ],
        };
        let clip_edges = vec![
            (
                0,
                Line {
                    start: Point { x: 0.75, y: 1.75 },
                    direction: Direction::new(1.0, -1.0),
                },
            ),
            (
                1,
                Line {
                    start: Point { x: 0.0, y: 0.0 },
                    direction: Direction::new(1.0, 1.0),
                },
            ),
        ];
        let site = Point { x: 1.0, y: 2.0 };

        let points = clip(&subject, &clip_edges, &site);

        let expected_points = vec![
            Point { x: 1.0, y: 2.0 },
            Point { x: 1.0, y: 1.5 },
            Point { x: 0.5, y: 2.0 },
        ];
        for ((point, _), expected_point) in points.iter().zip(expected_points.iter()) {
            assert!(point.close_to(expected_point, 0.001));
        }
    }
}
