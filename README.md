This is a repository for Chest X-Ray COVID-19 detection based on a COvidNet and different preprocessing steps. The network impolementation is based on the COVIDNet proposed in L. Wang, Z. Q. Lin, and A. Wong, “Covid-net: A tailored deep convolutional neural network design for detection of covid-19 cases from chest radiography images,” Scientific Reports, vol. 10, no. 19549, 2020, following the PyTorch implementation available at [GitHub](https://github.com/IliasPap/COVIDNet}.)

The respository includes also impelementation of a network based on ResNet50, as proposed in: Li, L., Qin, L., Xu, Z., Yin, Y., Wang, X., Kong, B., Bai, J., Lu, Y., Fang, Z., Song, Q. and Cao, K., 2020. Using artificial intelligence to detect COVID-19 and community-acquired pneumonia based on pulmonary CT: evaluation of the diagnostic accuracy. Radiology, 296(2). [url](https://pubs.rsna.org/doi/10.1148/radiol.2020200905?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed)

The models available are:

- Original CovidNet
- Modified CovidNet with dropout and Grad_CAM functionalities
- CovidNet with Deep Explainer functionalities
- DenseNet

The preprocessing steps evaluated include cropping and segmentation of the lungs based on the 2D Lung segmentation available at [GitHub](https://github.com/imlab-uiip/lung-segmentation-2d). This network is also included in the segmenter directory of this respository. The description of all the experiments and results can be consulted on:

J.D. Arias-Londoño, J.A. Gómez-García, L. Moro-Velázquez, J.I. Godino-Llorente. Artificial Intelligence applied to chest X-Ray images for the automatic detection of COVID-19. A thoughtful evaluation approach. In press IEEE Access, 2020. [Open access](https://ieeexplore.ieee.org/document/9293268)

If you find our work useful, can cite our paper using:

```
@article{arias2020,
  title={Artificial Intelligence applied to chest X-Ray images for the automatic detection of COVID-19. A thoughtful evaluation approach},
  author={Arias-Londo{\~n}o, Julian D and Gomez-Garcia, Jorge A and Moro-Velazquez, Laureano and Godino-Llorente, Juan I},
  journal={In Press IEEE Access},
  year={2020}
}
```

