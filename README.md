
# Real-Time-Traffic-Noise-Monitoring

## Project Overview

This project implements a real-time big data pipeline to monitor, analyze, and predict the impact of urban traffic noise on commuter stress. Using simulated IoT noise sensors placed on real road networks, the system combines streaming data processing, machine learning, and sentiment analysis to identify noisy areas, predict future conditions, and recommend quieter routes.

The system is designed as a modular, scalable architecture that demonstrates key big data concepts such as high-velocity ingestion, stream processing, predictive analytics, and spatial analysis.

System Architecture

The project follows a layered architecture:

Synthetic Data Generation – Geo-aware noise sensors simulated on real Milwaukee road segments

Streaming Ingestion – Apache Kafka for high-throughput, real-time data ingestion

Stream Processing – Apache Spark Structured Streaming for validation and aggregation

Persistent Storage – PostgreSQL for structured analytical storage

Analytics Services – Noise prediction, sentiment correlation, and stress zone detection

API & Visualization – REST APIs and Streamlit dashboards for user interaction

Key Features

Realistic noise sensor simulation bound to OpenStreetMap road geometry

Continuous streaming with Kafka and Spark Structured Streaming

Rolling-window noise aggregation and spatial segmentation

Machine learning–based noise prediction using Random Forest models

Noise–sentiment correlation analysis with statistical validation

Stress zone prediction combining noise forecasts and sentiment signals

Quiet-route recommendations based on real-time and predicted conditions

Technologies Used

Apache Kafka – Streaming data ingestion

Apache Spark Structured Streaming – Real-time processing

PostgreSQL – Data storage

Python – Core implementation

scikit-learn – Machine learning models

OSMnx / OpenStreetMap – Road network data

Flask – REST API services

Streamlit – User interface and visualization

Docker – Containerized deployment

Data Model

The database schema includes tables for:

Raw noise readings

Aggregated noise statistics

Route-based noise segments

Sentiment scores

Noise–sentiment correlations

Noise predictions

Current and predicted stress zones

Alternative routes and recommendations

<img width="1920" height="849" alt="image" src="https://github.com/user-attachments/assets/659dec7c-9d9d-4253-a5ca-8482846c6014" />
<img width="1920" height="858" alt="image" src="https://github.com/user-attachments/assets/48938e74-0d83-4342-aacb-4e3dae9b2763" />
<img width="1917" height="824" alt="image" src="https://github.com/user-attachments/assets/d58331a3-918f-408f-b846-1f1c5cd1c2d5" />
<img width="1919" height="799" alt="image" src="https://github.com/user-attachments/assets/956ead2a-74e1-471b-806b-33049705de83" />
<img width="1857" height="820" alt="image" src="https://github.com/user-attachments/assets/bf3d6771-76f4-40d8-b7d7-19498a7e3359" />
