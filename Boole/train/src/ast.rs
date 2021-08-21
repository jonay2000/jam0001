use crate::operations::Operation;
use crate::parse::parser::Span;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Program {
    pub trains: Vec<Train>,
    pub stations: Vec<Station>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Train {
    pub start: Target,
    pub first_class_passengers: Vec<FirstClassPassenger>,
    pub second_class_passengers: Vec<SecondClassPassenger>
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FirstClassPassenger {
    pub name: String,
    pub data: String,
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SecondClassPassenger {
    pub name: String,
    pub data: i64
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Station {
    pub name: String,
    pub operation: Operation,
    pub output: Vec<Target>
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Target {
    pub span: Span,
    pub station: String,
    pub track: usize,
}