use super::*;

fn get_expectation(expected_edges: Vec<((f32, f32), (f32, f32))>) -> Vec<LineSegment> {
    let mut expected = vec![];
    for (first, second) in expected_edges {
        expected.push(LineSegment {
            first: Point {
                x: first.0,
                y: first.1,
            },
            second: Point {
                x: second.0,
                y: second.1,
            },
        })
    }
    return expected;
}

fn check_line_segments_close(
    first_segments: Vec<(u32, LineSegment)>,
    second_segments: Vec<LineSegment>,
) {
    let mut sorted_first_segments = first_segments.clone();
    sorted_first_segments.sort_by_key(|(key, _)| *key);
    println!("sorted first segments: {:?}", sorted_first_segments);

    for (_, edge) in &sorted_first_segments {
        println!(
            "(({:.2}, {:.2}), ({:.2}, {:.2})), ",
            edge.first.x, edge.first.y, edge.second.x, edge.second.y
        );
    }

    assert_eq!(sorted_first_segments.len(), second_segments.len());
    for ((_, first_segment), second_segment) in
        sorted_first_segments.iter().zip(second_segments.iter())
    {
        assert!(first_segment.first.close_to(&second_segment.first, 0.01));
        assert!(first_segment.second.close_to(&second_segment.second, 0.01));
    }
}

#[test]
fn test_fortunes_split_arc_being_squeezed() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(-1.0, 0.0), (0.0, 1.0), (0.05, -1.0), (0.1, 0.1)];
    let expected_edges = vec![
        ((-1.00, 1.00), (-0.49, 0.49)),
        ((-1.00, 1.00), (-2.00, 2.00)),
        ((-0.95, -1.00), (-0.41, -0.43)),
        ((-0.95, -1.00), (-1.90, -2.00)),
        ((-0.45, 0.10), (-0.49, 0.49)),
        ((-0.45, 0.10), (-0.41, -0.43)),
        ((-0.49, 0.49), (2.00, 0.77)),
        ((-0.41, -0.43), (2.00, -0.54)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    println!("edges: {:?}", edges);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_8() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![
        (-1.26, 1.02),
        (0.36, 0.64),
        (0.83, 0.45),
        (-1.07, -1.07),
        (1.74, -1.05),
    ];
    let expected_edges = vec![
        ((-12.66, -1.07), (-0.64, 0.02)),
        ((-0.49, 0.64), (-0.64, 0.02)),
        ((-0.49, 0.64), (-0.18, 2.00)),
        ((-0.64, 0.02), (0.13, -0.62)),
        ((0.56, 0.45), (0.13, -0.62)),
        ((0.56, 0.45), (1.18, 2.00)),
        ((0.13, -0.62), (0.33, -0.88)),
        ((0.33, -1.05), (0.33, -0.88)),
        ((0.33, -1.05), (0.34, -2.00)),
        ((0.33, -0.88), (2.00, 0.13)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_7() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![
        (-1.77, 1.60),
        (0.24, 1.60),
        (1.14, -1.51),
        (-1.20, -0.76),
        (0.54, -1.08),
    ];
    let expected_edges = vec![
        ((-6.37, -0.76), (-0.76, 0.59)),
        ((-0.76, 1.60), (-0.76, 0.59)),
        ((-0.76, 1.60), (-0.76, 2.00)),
        ((-0.36, -1.08), (-0.12, 0.20)),
        ((-0.36, -1.08), (-0.83, -3.62)),
        ((-0.76, 0.59), (-0.12, 0.20)),
        ((0.69, -1.51), (-0.83, -3.62)),
        ((0.69, -1.51), (2.09, 0.45)),
        ((-0.12, 0.20), (2.09, 0.45)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_6() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![
        (-1.00, 1.97),
        (1.68, 0.86),
        (-0.92, -0.36),
        (0.98, -1.07),
        (0.32, -1.18),
    ];
    let expected_edges = vec![
        ((-34.89, -0.36), (0.10, 0.84)),
        ((-0.57, -1.18), (0.38, 0.25)),
        ((-0.57, -1.18), (-1.11, -2.00)),
        ((0.64, -1.07), (0.43, 0.22)),
        ((0.64, -1.07), (0.80, -2.00)),
        ((0.11, 0.86), (0.10, 0.84)),
        ((0.11, 0.86), (0.58, 2.00)),
        ((0.10, 0.84), (0.38, 0.25)),
        ((0.38, 0.25), (0.43, 0.22)),
        ((0.43, 0.22), (2.00, -0.35)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_5() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(-0.46, -0.94), (-0.28, 1.74), (-0.35, -0.10), (1.00, -1.10)];
    let expected_edges = vec![
        ((-3.61, -0.10), (-14.86, 1.37)),
        ((-3.61, -0.10), (0.31, -0.61)),
        ((-20.32, 1.74), (-14.86, 1.37)),
        ((-14.86, 1.37), (1.33, 0.76)),
        ((0.26, -1.10), (0.31, -0.61)),
        ((0.26, -1.10), (0.16, -2.00)),
        ((0.31, -0.61), (1.33, 0.76)),
        ((1.33, 0.76), (2.00, 1.06)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_4() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(-1.68, 0.22), (-1.06, -0.80), (-0.25, -0.40), (1.57, -1.90)];
    let expected_edges = vec![
        ((-2.21, -0.80), (-0.94, -0.03)),
        ((-0.75, -0.40), (-0.94, -0.03)),
        ((-0.75, -0.40), (0.01, -1.94)),
        ((-0.94, -0.03), (-0.06, 2.00)),
        ((0.04, -1.90), (0.01, -1.94)),
        ((0.04, -1.90), (2.00, 0.48)),
        ((0.01, -1.94), (-0.02, -2.00)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_3() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(-0.10, 1.57), (0.62, 1.48), (1.66, 0.43), (1.84, -0.19)];
    let expected_edges = vec![
        ((0.25, 1.48), (0.05, -0.12)),
        ((0.25, 1.48), (0.32, 2.00)),
        ((0.61, 0.43), (0.05, -0.12)),
        ((0.61, 0.43), (2.00, 1.81)),
        ((0.05, -0.12), (-0.15, -0.43)),
        ((0.68, -0.19), (-0.15, -0.43)),
        ((0.68, -0.19), (2.00, 0.19)),
        ((-0.15, -0.43), (-1.57, -2.00)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_2() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![
        (-1.77, 1.17),
        (-1.76, 1.89),
        (-1.57, -1.87),
        (1.70, 0.44),
        (0.32, -1.18),
    ];
    let expected_edges = vec![
        ((-27.69, 1.89), (0.11, 1.50)),
        ((-24.77, -1.87), (-1.07, -0.31)),
        ((-0.75, -1.18), (-1.07, -0.31)),
        ((-0.75, -1.18), (-0.45, -2.00)),
        ((-1.07, -0.31), (-0.09, 0.56)),
        ((0.06, 0.44), (-0.09, 0.56)),
        ((0.06, 0.44), (2.00, -1.21)),
        ((-0.09, 0.56), (0.11, 1.50)),
        ((0.11, 1.50), (0.32, 2.00)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_fail_bug_1() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(0.14, -1.10), (0.44, -0.08), (0.42, -1.09)];
    let expected_edges = vec![
        ((0.28, -1.09), (0.26, -0.58)),
        ((0.28, -1.09), (0.31, -2.00)),
        ((-1.44, -0.08), (0.26, -0.58)),
        ((-1.44, -0.08), (-2.00, 0.08)),
        ((0.26, -0.58), (2.00, -0.62)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_edge_bug_1() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(-1.90, 1.00), (1.00, 0.00), (1.80, 1.30)];
    let expected_edges = vec![
        ((-0.62, 0.00), (-0.08, 1.56)),
        ((-0.62, 0.00), (-1.31, -2.00)),
        ((0.34, 1.30), (-0.08, 1.56)),
        ((0.34, 1.30), (2.00, 0.28)),
        ((-0.08, 1.56), (-0.12, 2.00)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_edge_bug_2() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(0.00, 0.20), (0.50, 1.50), (0.70, 0.10)];
    let expected_edges = vec![
        ((-1.44, 1.50), (0.44, 0.78)),
        ((-1.44, 1.50), (-2.00, 1.72)),
        ((0.34, 0.10), (0.44, 0.78)),
        ((0.34, 0.10), (0.04, -2.00)),
        ((0.44, 0.78), (2.00, 1.00)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_x_line() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(0.00, 0.00), (1.00, 0.00), (2.00, 0.00)];
    let expected_edges = vec![
        ((0.50, 0.00), (0.50, 2.00)),
        ((0.50, 0.00), (0.50, -2.00)),
        ((1.50, 0.00), (1.50, 2.00)),
        ((1.50, 0.00), (1.50, -2.00)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}

#[test]
fn test_fortunes_right_opening_v() {
    let boundary_polyline = vec![(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0)];
    let sites = vec![(0.00, 0.00), (1.00, 1.00), (1.01, -1.00)];
    let expected_edges = vec![
        ((0.00, 1.00), (1.00, -0.00)),
        ((0.00, 1.00), (-1.00, 2.00)),
        ((0.01, -1.00), (1.00, -0.00)),
        ((0.01, -1.00), (-0.98, -2.00)),
        ((1.00, -0.00), (2.00, 0.00)),
    ];

    let edges = run_fortunes(sites, boundary_polyline);
    let expected = get_expectation(expected_edges);
    check_line_segments_close(edges, expected);
}
