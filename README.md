# RGB2MultiSpectral

# 🌍 Hyperspectral Image Cube Generation Pipeline

## 🔗 Data Access & Requirements

1. **Download Input Data**  
   Please add the shared Drive directory (containing all required input data such as materials and satellite images) into the root folder of the pipeline.

2. **Drive Link**  
   [📁 Google Drive Folder – Input Data](https://drive.google.com/drive/folders/1UkzIlgpd1S2eq6uj-W6LJEsrJJDIuGLh?usp=sharing)  
   *Note: This link will expire after one year. If access is needed after that, please contact the paper's author.*

---

## 📁 Folder Structure & Descriptions

### `results/sentilent_data/patches_anno/`
Contains examples of cropped frames and their corresponding annotations. This includes:
- Input satellite images
- JSON annotation files
- Output masks derived from the annotations  

📌 Use these files to test several functions in the `DataGenerator_dev` module.

---

### `results/sentilent_data/patches_after_anno/`
Augmented samples generated after annotating the cropped Sentinel-2 data.

---

### `results/sentilent_data/metrics/`
Includes evaluation metrics for clustering RGB values across all segments of the dataset.

---

### `results/sentilent_data/full_dataset/`
Due to limited Drive storage, only test input data is included.  
To obtain the full dataset (train/val/test):
1. Contact the author  
2. Use the `DataGenerator_dev` class along with the data in `results/sentilent_data/full_data/` to regenerate the complete dataset locally.

---

### `results/cubes_generator/Max_score/` and `results/cubes_generator/raffle_score/`
These folders contain:
- Examples of pipeline results
- Associated evaluation metrics (based on 1,000 samples)

📌 To get full results:
- Run the pipeline on `results/sentilent_data/full_dataset/test/`, or  
- Contact the author to access the complete output set.

---

## ⚙️ Pipeline Usage Guidelines

- Follow the intended execution order of functions in each module/package to ensure the pipeline runs correctly.
- In the `back/` package, there's no need to re-train the semantic segmentation model unless working with new data.
- Use the `DataGenerator` class to inject your own dataset for:
  - Training a new model
  - Testing on unseen data
- The `HVI-CIDNet` and `HVI-low_to_high_light` networks were trained using the `generalization.pth` file. These should be applied to the `train/val/test` folders inside `results/sentilent_data/full_dataset/`.

---

## 🚀 Recommendation

To achieve the best performance, apply the pipeline directly on your own input data using the full process described above.

---

## 📄 Citation

If you use this project in your research or publication, please cite the following:

```bibtex
@article{yan2025hvi,
  title={HVI: A New color space for Low-light Image Enhancement},
  author={Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Pang, Guansong and Shi, Kangbiao and Wu, Peng and Dong, Wei and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2502.20272},
  year={2025}
}


1. add data from drive shared directory with all input data for the project (materials + satellite images) to pipeline main folder.
2. link to Drive files: https://drive.google.com/drive/folders/1UkzIlgpd1S2eq6uj-W6LJEsrJJDIuGLh?usp=sharing (will be deleted after a year and if the data is needed you can contact the paper author).
3. on results/sentilent_data/patches_anno there is example for several frames and their annotations to run several function in DataGenerator_dev. (images input , json of annotations data and corresponde output mask of json).
4. on results/sentilent_data/patches_after_anno - several exmaples of augmanted data after annotating the croped sentilent data.
5. on results/sentilent_data/metrics - shows metrics for clustering RGB values on all segments of the dataset.
6. on results/sentilent_data/full_dataset - found only test input data for the pipeline due to lack of drive space. for getting full dataset (train,val, test):1. reach author, 2.run functions from DataGenerator_dev class and the data from results/sentilent_data/full_data will generate all the set locally on your PC.
7. on results/cubes_generator/Max_score // results/cubes_generator/raffle_score you can find several examples for pipeline results + metrics (metrics are based on full pipeline output - 1000 samples). if you like to get full results for all dataset you can run the pipeline on results/sentilent_data/full_dataset/test and get all the 1000 samples results to your own PC or contact the author.
8. on each package and each mudule you should run usecase by the functions order to let the pipeline work correctly.
9. on back package there is no need to run trainning session for semantic NN (only if you like to train on new data).
10. Datagenerator class is used if you would like to insert you own data for traning and testing sessions (net training and testing pipeline based on the trained net on new data)
11. HVI-CIDNet, HVI-low_to_high_light was used with use generalization.pth after splitting the data to train, val and test bases - you should run it on the mentioned folders from results/sentilent_data/full_dataset.
12. for best pipeline result apply the project on your input data images.
13. @article{yan2025hvi,
  title={HVI: A New color space for Low-light Image Enhancement},
  author={Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Pang, Guansong and Shi, Kangbiao and Wu, Peng and Dong, Wei and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2502.20272},
  year={2025}
}

