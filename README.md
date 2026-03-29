# Eye Tracking System

## Overview
This project is a simple tool to test and measure eye-tracking performance.  
It helps compare how accurate and stable different eye-tracking setups are.

---

## Features
- Measures accuracy and precision  
- Supports webcam and CSV data  
- Includes basic tasks:
  - Grid tracking  
  - Fixation  
  - Moving target tracking  
- Exports results in JSON, CSV, and HTML  

---

## Project Structure
src/ → main code (tasks, metrics, devices)
tests/ → test files
data/ → sample data and outputs
configs/ → settings files
scripts/ → run scripts


---

## Installation

1. Clone the repository:
   git clone https://github.com/shukti7443/eye-tracking-benchmark.git
cd eye-tracking-benchmark


2. Create environment:
   python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows

3. Install dependencies:
pip install -r requirements.txt

---

## Quick Start

### Run with webcam
python scripts/run_benchmark.py --device webcam --task grid

### Run with CSV data
python scripts/run_benchmark.py --device csv --input sample.csv

---

## Tasks
- Grid Task → checks accuracy on fixed points  
- Fixation Task → checks stability  
- Pursuit Task → tracks moving object  
- Saccade Task → measures fast eye movement  

---

## Metrics
- Accuracy error  
- Precision (stability)  
- Data loss  
- RMS error  

---

## Output
Results are saved in:
data/exports/

Includes:
- JSON report  
- CSV data  
- HTML summary  

---

## Run Tests
pytest tests/

---

## Contributing
- Fork the repository  
- Create a new branch  
- Make changes  
- Submit a pull request  
