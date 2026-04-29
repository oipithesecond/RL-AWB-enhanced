# RL-AWB-enhanced

This repository contains an extended version of the RL-AWB framework for nighttime color constancy and image enhancement. It introduces a hand-crafted image signal processing (ISP) pipeline and a hybrid no-reference reward system to produce natural and visually rich images.

## Key Features & Contributions

* **Custom ISP Pipeline**: Incorporates gamma correction, contrast stretching, and saturation control directly into the agent's environment.
* **Expanded Action Space**: The Soft Actor-Critic (SAC) agent utilizes a 5-dimensional continuous action space (modifying gamma, alpha, and gain alongside traditional white-balance parameters).
* **Perceptual Quality Rewards**: A hybrid reward system utilizing the MUSIQ perceptual metric at episode termination and BRISQUE as a fast mid-episode proxy.
* **Zero Calibration Targets**: The enhancements allow the agent to produce high-quality images without requiring a physical color chart in the scene.

## Architecture Overview

1. **White Balance Engine (SGP-LRD)**: Detects reliable gray pixels and extracts features to build a state representation for the RL agent.
2. **Soft Actor-Critic (SAC) Agent**: Receives an image-level state observation (RGB-UV histogram + action history) and outputs a 5-dimensional vector to dynamically select parameters.
3. **ISP Downstream Tuning**: Applies exposure/shadow lift via Gamma, contrast distribution via Alpha, and color vibrancy via Saturation Gain. Includes specific penalties for asymmetric clipping and color variance to prevent artifacts.

## Installation

1. Clone this repository:

       git clone https://github.com/adityaxgupta/rl-awb-enhanced.git
       cd rl-awb-enhanced

2. Install the required dependencies:

       pip install -r requirements.txt

3. **Model Weights**: Download the pre-trained SAC model (rl_awb_stage1_true_final.zip) from the Releases section of this repository and place it in the root directory.

## Usage

You can test the system on your own nighttime images using the provided Gradio interface.

1. Launch the application:

       python app.py

2. Open the provided local URL in your browser.
3. Upload any dark, nighttime photograph.
4. The agent will process the image, automatically predicting the optimal gamma, contrast, and saturation levels, and display the enhanced output alongside the AI-selected parameters.

## Repository Structure

* app.py: Main Gradio web application for inference.
* isp_pipeline.py: Contains the custom image signal processing functions and the asymmetric clipping penalties.
* utils/WBsRGB.py: Utilities for RGB-UV histogram feature extraction.
