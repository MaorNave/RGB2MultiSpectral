# RGB2MultiSpectral

1. add data from drive shared directory with all input data for the project (materials + satellite images)
to pipeline main folder.
2. on each package and each mudule you should run usecase by the functions order to let the pipeline work correctly.
3. on back package there is no need to run trainning session for semantic NN (only if you like to train on new data).
4. Datagenerator class is used if you would like to insert you own data for traning and testing sessions (net training and testing pipeline based on the trained net on new data)
5. HVI-CIDNet, HVI-low_to_high_light was used with  use generalization.pth after splitting the data to train, val and test bases.
6. for best pipeline result apply the project on your input data images.
7. @article{yan2025hvi,
  title={HVI: A New color space for Low-light Image Enhancement},
  author={Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Pang, Guansong and Shi, Kangbiao and Wu, Peng and Dong, Wei and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2502.20272},
  year={2025}
}
