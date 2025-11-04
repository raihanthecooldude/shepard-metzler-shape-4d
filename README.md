# 4D Shepard-Metzler Shape Experiment

This 4D Shepard-Metzler Shape Experiment is a custom-developed 4D metzler shaped object with python-openGL by generating perspective pictures of four-dimensional abstract objects for studying mental rotation done by [Shepard R.N. & Metzler J.(1971)](https://www.science.org/doi/10.1126/science.171.3972.701).

<div align="center">
  <img src="./media/metzler-shape-4d.png">
</div>

## Requirement

```console
$ python>=3.13
$ matplotlib==3.9
$ moviepy==1.0.3
```

## Folder Structure

```
src
|
|---> exp-setup: main experimental setup for generating images and code to conduct the experiment (additional: the main rotating 4D metzler shape)
|
|---> opaque: rendering and projection code for different stages of opaque 4D tesseracts
|
|---> transparent: rendering and projection code for different stages of transparent 4D tesseracts
```

## Run the project
if you use conda:
```
conda create -n myenv python=3.13
```
```
conda activate myenv
```
```
conda install numpy matplotlib==3.9.2 joblib ipywidgets
pip install moviepy==1.0.3 glfw pyopengl
```