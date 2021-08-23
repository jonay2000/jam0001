use train::ast::{Program, Target};
use std::path::PathBuf;
use crate::frontend::web::runner::GenerateVisualizerDataError;
use train::operations::Operation;
use serde::*;
use std::fs::File;
use std::io::Write;
use rand::{Rng, thread_rng};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use strum::*;
use pathfinding::directed::astar::astar;
use crate::frontend::web::layout::Direction::SOUTH;
use noise::{OpenSimplex, NoiseFn, Seedable};
use rand::prelude::*;

pub async fn layout_to_file(program: &Program, connection_id: String) -> Result<PathBuf, GenerateVisualizerDataError> {
    let res = layout(program).await?;
    let serialized = serde_json::to_string_pretty(&res)?;

    let mut path = PathBuf::new();
    path.push("visualizer");
    path.push("visualizer_setup");
    path.push(format!("{}.json.result.json", connection_id));
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    let mut file = File::create(&path)?;
    file.write_all(serialized.as_bytes())?;
    file.flush()?;

    let mut path = PathBuf::new();
    path.push("visualizer_setup");
    path.push(format!("{}.json.result.json", connection_id));

    Ok(path)
}

pub async fn layout(program: &Program) -> Result<LayoutResult, GenerateVisualizerDataError> {
    let mut stations = layout_stations(program)?;
    let lines = layout_lines(program, &mut stations)?;
    let tiles = layout_tiles(program, &mut stations, &lines)?;
    Ok(LayoutResult {
        stations,
        lines,
        tiles,
    })
}

pub fn layout_stations(program: &Program) -> Result<Vec<LayoutStation>, GenerateVisualizerDataError> {
    log::debug!("Layout out stations...");
    let mut rng = rand::thread_rng();

    let mut station_locs: Vec<(f32, f32)> = program.stations.iter().map(|s| (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))).collect();
    let station_conns: Vec<Vec<bool>> = program.stations.iter().map(|s1| {
        program.stations.iter().map(|s2| {
            s1.output.iter().any(|t1| t1.station == s2.name) ||
                s2.output.iter().any(|t2| t2.station == s1.name)
        }).collect()
    }).collect();

    const REPULSIVE_CONSTANT : f32 = 50.0;
    const ATTRACTIVE_CONSTANT : f32 = 0.1;
    const RANDOM_CONSTANT : f32 = 0.01;

    let mut forces: Vec<(f32, f32)> = Vec::with_capacity(program.stations.len());
    let mut iter_count = 0;
    for i in 0.. {
        for (s1i, &(s1x, s1y)) in station_locs.iter().enumerate() {
            let final_force = station_locs.iter().enumerate().map(|(s2i, &(s2x, s2y))| {
                if s1i == s2i {return (0.0, 0.0)}

                let d = (s1x-s2x).abs() + (s1y-s2y).abs();
                let d = if d == 0.0 {0.1} else {d};

                // Repulsive force
                let rep_force = -1.0 / ((d*d) / REPULSIVE_CONSTANT);

                // Attractive force
                let att_force = if station_conns[s1i][s2i] {
                    d * ATTRACTIVE_CONSTANT
                } else {
                    0.0
                };

                // Random force
                let rand_force = rng.gen_range(-RANDOM_CONSTANT..RANDOM_CONSTANT);

                let force = rep_force + att_force + rand_force;
                ((s2x-s1x)/d*force, (s2y-s1y)/d*force)
            }).fold((0.0, 0.0), |(ax, ay), (bx, by)| (ax+bx, ay+by));
            forces.push(final_force)
        }

        for (s1i, (s1x, s1y)) in station_locs.iter_mut().enumerate() {
            *s1x += forces[s1i].0;
            *s1y += forces[s1i].1;
        }

        if i > 100 && forces.iter().map(|(fx, fy)| fx + fy).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() < 0.05 {
            break;
        }
        if i % 100 == 0 {
            log::debug!("Did {} iterations of station layout", i);
        }
        iter_count = i;

        forces.clear();
    }

    log::debug!("Finished station layout in {} iterations.", iter_count);

    Ok(program.stations.iter().zip(station_locs.into_iter()).map(|(s, (x, y))| {
        LayoutStation {
            x: x.round() as isize,
            y: y.round() as isize,
            stoppers: vec![false; 8],
            typ: s.operation,
            name: s.name.clone()
        }
    }).collect())
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, EnumIter)]
enum Direction {
    NORTH, EAST, SOUTH, WEST
}
impl Direction {
    pub fn vector(&self) -> (isize, isize) {
        match &self {
            Direction::NORTH => (0, -1),
            Direction::EAST => (1, 0),
            Direction::SOUTH => (0, 1),
            Direction::WEST => (-1, 0)
        }
    }

    pub fn from_vector(vec: (isize, isize)) -> Self {
        match vec {
            (0, -1) => Direction::NORTH,
            (1, 0) => Direction::EAST,
            (0, 1) => Direction::SOUTH,
            (-1, 0) => Direction::WEST,
            _ => unreachable!()
        }
    }

    pub fn to_id(&self) -> usize {
        match &self {
            Direction::NORTH => 0,
            Direction::EAST => 1,
            Direction::SOUTH => 2,
            Direction::WEST => 3
        }
    }

    pub fn inv(&self) -> Self {
        match &self {
            Direction::NORTH => Direction::SOUTH,
            Direction::EAST => Direction::WEST,
            Direction::SOUTH => Direction::NORTH,
            Direction::WEST => Direction::EAST
        }
    }
}

pub fn layout_lines(program: &Program, stations: &mut Vec<LayoutStation>) -> Result<Vec<LayoutLine>, GenerateVisualizerDataError> {
    const PADDING: isize = 20;
    let grid_min = stations.iter().map(|s| s.x.min(s.y)).min().unwrap() - PADDING;
    let grid_max = stations.iter().map(|s| s.x.max(s.y)).max().unwrap() + PADDING;
    let size = (grid_max - grid_min + 1) as usize;

    let mut grid = vec![vec![LineTile::Empty; size]; size];

    for station in stations.iter() {
        grid[(station.x - grid_min) as usize][(station.y - grid_min) as usize] = LineTile::Station;
        grid[(station.x + 1 - grid_min) as usize][(station.y - grid_min) as usize] = LineTile::Station;
        grid[(station.x - grid_min) as usize][(station.y + 1 - grid_min) as usize] = LineTile::Station;
        grid[(station.x + 1 - grid_min) as usize][(station.y + 1 - grid_min) as usize] = LineTile::Station;
    }

    let mut lines = Vec::with_capacity(stations.len() * 2);
    for (from_id, station) in program.stations.iter().enumerate() {
        for (station_track, target) in station.output.iter().enumerate() {
            let from = vec![
                (stations[from_id].x, stations[from_id].y),
                (stations[from_id].x + 1, stations[from_id].y),
                (stations[from_id].x, stations[from_id].y + 1),
                (stations[from_id].x + 1, stations[from_id].y + 1),
            ].into_iter().map(|(a, b)| ((a-grid_min) as usize, (b-grid_min) as usize)).collect();
            let to_id = stations.iter().enumerate().find(|(i, s)| s.name == target.station).unwrap().0;
            let to = vec![
                (stations[to_id].x, stations[to_id].y),
                (stations[to_id].x + 1, stations[to_id].y),
                (stations[to_id].x, stations[to_id].y + 1),
                (stations[to_id].x + 1, stations[to_id].y + 1),
            ].into_iter().map(|(a, b)| ((a-grid_min) as usize, (b-grid_min) as usize)).collect();
            let path: Vec<_> = run_astar(&mut grid, from, to, to_id, target.track)?.into_iter().map(|(a, b)| (a as isize + grid_min, b as isize + grid_min)).collect();
            lines.push(LayoutLine {
                station_id: from_id, station_track, path
            })
        }
    }

    Ok(lines)
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum LineTile {
    Station,
    Track(usize, usize),
    Empty
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
enum LinePoint {
    Start,
    Regular((usize, usize), Option<Direction>),
    MidCross((usize, usize), Direction)
}

impl LinePoint {
    pub fn tuple(self) -> (usize, usize) {
        match self {
            LinePoint::Start => unreachable!(),
            LinePoint::Regular(n, _) => n,
            LinePoint::MidCross(n, _) => n,
        }
    }
}

fn run_astar(grid: &mut Vec<Vec<LineTile>>, from: Vec<(usize, usize)>, to: Vec<(usize, usize)>, station: usize, station_track: usize)
             -> Result<Vec<(usize, usize)>, GenerateVisualizerDataError> {
    let move_dir = |loc: (usize, usize), dir: Direction| -> Option<(usize, usize)> {
        let nb = ((loc.0 as isize) + dir.vector().0, (loc.1 as isize) + dir.vector().1);
        if nb.0 < 0 || nb.1 < 0 || nb.0 >= grid.len() as isize || nb.1 >= grid.len() as isize {
            None
        } else {
            Some((nb.0 as usize, nb.1 as usize))
        }
    };
    let mut path = astar(
        &LinePoint::Start,
        |n| {
            match n {
                LinePoint::Start => {
                    from.iter().map(|&l| (LinePoint::Regular(l, None), 0)).collect()
                },
                LinePoint::Regular(n, straight) => {
                    Direction::iter().map(|dir: Direction| {
                        match move_dir(*n, dir) {
                            None => vec![],
                            Some(nb) => match grid[nb.0][nb.1] {
                                LineTile::Station => {
                                    if to.contains(&nb) {
                                        vec![(LinePoint::Regular(nb, Some(dir)), 10)]
                                    }else if from.contains(n) && from.contains(&nb) {
                                        vec![(LinePoint::Regular(nb, Some(dir)), 0)]
                                    } else {
                                        vec![]
                                    }
                                }
                                LineTile::Track(to_stat, to_stat_track) => {
                                    if station == to_stat && station_track == to_stat_track {
                                        //We can join
                                        vec![(LinePoint::Regular(nb, Some(dir)), 0)]
                                    } else {
                                        //We can (try to) cross
                                        vec![(LinePoint::MidCross(nb, dir), 50)]
                                    }
                                },
                                LineTile::Empty => {
                                    if let Some(straight) = straight {
                                        if *straight == dir {
                                            vec![(LinePoint::Regular(nb, Some(dir)), 9)]
                                        } else {
                                            vec![(LinePoint::Regular(nb, Some(dir)), 10)]
                                        }
                                    }else {
                                        vec![(LinePoint::Regular(nb, Some(dir)), 10)]
                                    }
                                },
                                _ => unreachable!(),
                            }
                        }
                    }).flatten().collect::<Vec<_>>()
                }
                LinePoint::MidCross(n, dir) => {
                    match move_dir(*n, *dir) {
                        None => vec![],
                        Some(nb) => match grid[nb.0][nb.1] {
                            LineTile::Station => {
                                if to.contains(&nb) {
                                    vec![(LinePoint::Regular(nb, Some(*dir)), 10)]
                                } else {
                                    vec![]
                                }
                            }
                            LineTile::Track(_, _) => {
                                //Can only cross a single track
                                vec![]
                            },
                            LineTile::Empty => vec![(LinePoint::Regular(nb, Some(*dir)), 10)],
                            _ => unreachable!(),
                        }
                    }
                }
            }
        },
        |n| 0,
        |n| match n {
            LinePoint::Start => false,
            LinePoint::Regular(n, _) => to.contains(n),
            LinePoint::MidCross(n, d) => false,
        },
    ).ok_or(GenerateVisualizerDataError::GenerateVisualizerData)?.0;
    path.remove(0); //Remove start node
    if path.len() > 2 {
        for point in &path.as_slice()[1..path.len() - 1] {
            grid[point.tuple().0][point.tuple().1] = LineTile::Track(station, station_track);
        }
    }
    Ok(path.into_iter().map(|p| p.tuple()).collect())
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum VisualTile {
    Station,
    Track([bool; 4]), //NESW
    Empty,

    Decoration1,
    Decoration2,
    Decoration3,
    Decoration4,
    Decoration5,
    Water1,
    Water2,
    WaterLily,
    WaterSand,
}

impl VisualTile {
    pub fn to_tile_type(self) -> Option<&'static str> {
        match self {
            VisualTile::Station => None,
            VisualTile::Track(conns) => Some(match conns {
                [true, true, true, true] => "CROSSING",
                [true, false, true, false] => "Vertical",
                [false, true, false, true] => "Horizontal",
                [true, true, false, false] => "NE",
                [false, true, true, false] => "ES",
                [false, false, true, true] => "SW",
                [true, false, false, true] => "WN",
                [true, true, true, false] => "T_EAST",
                [true, true, false, true] => "T_NORTH",
                [true, false, true, true] => "T_WEST",
                [false, true, true, true] => "T_SOUTH",
                _ => unreachable!(),
            }),
            VisualTile::Empty => None,
            VisualTile::Decoration1 => Some("Decoration1"),
            VisualTile::Decoration2 => Some("Decoration2"),
            VisualTile::Decoration3 => Some("Decoration3"),
            VisualTile::Decoration4 => Some("Decoration4"),
            VisualTile::Decoration5 => Some("Decoration5"),
            VisualTile::Water1 => Some("water1"),
            VisualTile::Water2 => Some("water2"),
            VisualTile::WaterLily => Some("water_lily"),
            VisualTile::WaterSand => Some("water_sand"),
        }
    }
}


pub fn layout_tiles(program: &Program, stations: &mut Vec<LayoutStation>, lines: &Vec<LayoutLine>) -> Result<Vec<LayoutTile>, GenerateVisualizerDataError> {
    const PADDING: isize = 20;
    let grid_min = stations.iter().map(|s| s.x.min(s.y)).min().unwrap() - PADDING;
    let grid_max = stations.iter().map(|s| s.x.max(s.y)).max().unwrap() + PADDING;
    let size = (grid_max - grid_min + 1) as usize;
    let mut grid = vec![vec![VisualTile::Empty; size]; size];

    //Layout paths
    for line in lines {
        for wd in line.path.windows(2) {
            let dir = Direction::from_vector((wd[1].0 - wd[0].0, wd[1].1 - wd[0].1));

            let from_tile = &mut grid[(wd[0].0 - grid_min) as usize][(wd[0].1 - grid_min) as usize];
            if let VisualTile::Empty = from_tile {
                *from_tile = VisualTile::Track([false; 4]);
            }
            if let VisualTile::Track(arr) = from_tile {
                arr[dir.to_id()] = true;
            } else {
                unreachable!();
            }

            let to_tile = &mut grid[(wd[1].0 - grid_min) as usize][(wd[1].1 - grid_min) as usize];
            if let VisualTile::Empty = to_tile {
                *to_tile = VisualTile::Track([false; 4]);
            }
            if let VisualTile::Track(arr) = to_tile {
                arr[dir.inv().to_id()] = true;
            } else {
                unreachable!();
            }
        }
    }

    for station in stations.iter_mut() {
        station.stoppers[0] = if let VisualTile::Track(arr) = grid[(station.x - grid_min) as usize][(station.y - grid_min + 1) as usize] { arr[2] } else {false};
        station.stoppers[1] = if let VisualTile::Track(arr) = grid[(station.x - grid_min + 1) as usize][(station.y - grid_min + 1) as usize] { arr[2] } else {false};
        station.stoppers[2] = if let VisualTile::Track(arr) = grid[(station.x - grid_min + 1) as usize][(station.y - grid_min + 1) as usize] { arr[1] } else {false};
        station.stoppers[3] = if let VisualTile::Track(arr) = grid[(station.x - grid_min + 1) as usize][(station.y - grid_min) as usize] { arr[1] } else {false};
        station.stoppers[4] = if let VisualTile::Track(arr) = grid[(station.x - grid_min + 1) as usize][(station.y - grid_min) as usize] { arr[0] } else {false};
        station.stoppers[5] = if let VisualTile::Track(arr) = grid[(station.x - grid_min) as usize][(station.y - grid_min) as usize] { arr[0] } else {false};
        station.stoppers[6] = if let VisualTile::Track(arr) = grid[(station.x - grid_min) as usize][(station.y - grid_min) as usize] { arr[3] } else {false};
        station.stoppers[7] = if let VisualTile::Track(arr) = grid[(station.x - grid_min) as usize][(station.y - grid_min + 1) as usize] { arr[3] } else {false};
    }

    for station in stations.iter() {
        grid[(station.x - grid_min) as usize][(station.y - grid_min) as usize] = VisualTile::Station;
        grid[(station.x + 1 - grid_min) as usize][(station.y - grid_min) as usize] = VisualTile::Station;
        grid[(station.x - grid_min) as usize][(station.y + 1 - grid_min) as usize] = VisualTile::Station;
        grid[(station.x + 1 - grid_min) as usize][(station.y + 1 - grid_min) as usize] = VisualTile::Station;
    }

    add_details(&mut grid);



    let mut tiles = vec![];
    for (x, row) in grid.iter().enumerate() {
        for (y, tile) in row.iter().enumerate() {
            if let Some(str) = tile.to_tile_type() {
                tiles.push(LayoutTile {
                    x: (x as isize) + grid_min, y: (y as isize) + grid_min, typ: str
                })
            }
        }
    }

    Ok(tiles)
}

fn add_details(grid: &mut Vec<Vec<VisualTile>>) {
    let mut rng = thread_rng();

    let wnoise = OpenSimplex::new();
    wnoise.set_seed(rng.next_u32());
    let dnoise = OpenSimplex::new();
    dnoise.set_seed(rng.next_u32());
    let snoise = OpenSimplex::new();
    snoise.set_seed(rng.next_u32());

    for x in 0..grid.len() {
        for y in 0..grid.len() {
            if grid[x][y] != VisualTile::Empty { continue }

            //Water
            let nv = wnoise.get([(x as f64)/5.0, (y as f64)/5.0]);
            if nv > 0.3 {
                grid[x][y] = *[VisualTile::Water1, VisualTile::Water1, VisualTile::Water2, VisualTile::Water2, VisualTile::WaterLily].choose(&mut rng).unwrap();
            } else if nv > 0.25 {
                grid[x][y] = VisualTile::WaterSand;
            }

            //Decoration
            let nv = dnoise.get([(x as f64), (y as f64)]);
            if nv > 0.27 {
                grid[x][y] = *[VisualTile::Decoration1, VisualTile::Decoration2, VisualTile::Decoration3].choose(&mut rng).unwrap();
            } else if nv > 0.2 {
                grid[x][y] = *[VisualTile::Decoration4, VisualTile::Decoration4, VisualTile::Decoration5].choose(&mut rng).unwrap();
            }

            //More stones/grass
            let nv = snoise.get([(x as f64)/2.0, (y as f64)/2.0]);
            if nv > 0.3 {
                grid[x][y] = *[VisualTile::Decoration4, VisualTile::Decoration5].choose(&mut rng).unwrap();
            }
        }
    }

}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct LayoutResult {
    stations: Vec<LayoutStation>,
    lines: Vec<LayoutLine>,
    tiles: Vec<LayoutTile>,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct LayoutStation {
    x: isize,
    y: isize,
    stoppers: Vec<bool>,
    #[serde(rename="type")]
    typ: Operation,
    name: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct LayoutLine {
    station_id: usize,
    station_track: usize,
    path: Vec<(isize, isize)>
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct LayoutTile {
    x: isize,
    y: isize,
    #[serde(rename="type")]
    typ: &'static str,
}