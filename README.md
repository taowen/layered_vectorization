# Layered Image Vectorization via Semantic Simplification

<a href="https://szuviz.github.io/layered_vectorization/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2305.18203"><img src="https://img.shields.io/badge/arXiv-2305.16311-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>
<!-- Official implementation. -->
<br>


<p align="center">
<img src="/gallery_threerow.jpg" width="90%"/>  
  
<a href="#"> Zhenyu Wang</a>,
<a href="#"> Jianxi Huang</a> ,
<a href="https://zhdsun.github.io/">Zhida Sun</a>,
<a href="https://danielcohenor.com/">Daniel Cohen-Or</a>,
<a href="https://deardeer.github.io/">Min Lu</a>
> <br>
> This work presents a novel progressive image vectorization technique aimed
            at generating layered vectors that represent the original image from coarse
            to fine detail levels. Our approach introduces semantic simplification, which
            combines Score Distillation Sampling and semantic segmentation to iteratively simplify the input image. Subsequently, our method optimizes the
            vector layers for each of the progressively simplified images. Our method provides robust optimization, which avoids local minima and enables adjustable
            detail levels in the final output. The layered, compact vector representation enhances usability for further editing and modification. Comparative
            analysis with conventional vectorization methods demonstrates our technique’s superiority in producing vectors with high visual fidelity, and more
            importantly, maintaining vector compactness and manageability.
</p>

## Code (coming soon)
## Citation
If you find this useful for your research, please cite the following:
```bibtex
@article{vinker2023concept,
  title={Concept Decomposition for Visual Exploration and Inspiration},
  author={Yael Vinker and Andrey Voynov and Daniel Cohen-Or and Ariel Shamir},
  journal={arXiv preprint arXiv:2305.18203},
  year={2023}
}
```

