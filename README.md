# ICNoduleNet
This paper has been accepted for publication in JBHI 2024. And our code is based on [SANet](https://github.com/mj129/SANet), [NoduleNet](https://github.com/uci-cbcl/NoduleNet), and [SGDA](https://github.com/Ruixxxx/SGDA).

This code is licensed for non-commercial research purposes only.

## Contributions

We propose a multi-resolution pulmonary nodule dataset, named RKPN, with CT images acquired using both smooth and sharp kernels. It contains pairs of images taken from the same patient on the same date, providing the basis for analyzing the impact of reconstruction kernels on pulmonary nodule detection.

We quantify the performance of pulmonary nodule detectors using sharp reconstruction kernel images compared with those using smooth reconstruction kernel images.

We propose an ICNoduleNet to enhance the detection performance on sharp kernel imaging. It mainly contains a lightweight 3D slice-channel converter and a Charbonnier loss function.

## Datasets
Our dataset contains 292 pairs of annotated pulmonary nodule detection bounding boxes from two different reconstruction kernel imaging modalities. The example illustrates that pulmonary nodules in (B) sharp kernel imaging appear clearer compared to (A) smooth kernel imaging. 

<img width="935" height="597" alt="dataset" src="https://github.com/user-attachments/assets/66b89aa3-d6db-43fc-98ac-272319983f78" />

## Method
Overall architecture of the proposed ICNoduleNet. ICNoduleNet is an end-to-end neural network with three multi-task branches for image conversion, pulmonary nodule detection, and false positive reduction, respectively.

<img width="1465" height="824" alt="method" src="https://github.com/user-attachments/assets/e24a7e4a-e75d-4a39-a697-977def9af4ae" />

## Citations
If you are using the code/model/data provided here in a publication, please consider citing:
<details> <summary><b>
  @article{lan2024icnodulenet,
  title={Icnodulenet: Enhancing pulmonary nodule detection performance on sharp kernel ct imaging},
  author={Lan, Tianzhong and Zeng, Fanxin and Yi, Zhang and Xu, Xiuyuan and Zhu, Min},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={28},
  number={8},
  pages={4751--4760},
  year={2024},
  publisher={IEEE}
}
</b></summary>



## Contact
For any questions, please contact me via e-mail: lantianzhong1@stu.scu.edu.cn.
