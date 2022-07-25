---
layout: default
---

# Shadow Art Revisited: A Differentiable Rendering Based Approach
<a href="https://kaustubh-sadekar.github.io/" target="_blank">Kaustubh Sadekar</a>, <a href="https://www.linkedin.com/in/ashish-tiwari-82a392135/" target="_blank">Ashish Tiwari</a>, <a href="https://people.iitgn.ac.in/~shanmuga/index.html" target="_blank">Shanmuganathan Raman</a>

<a href="https://arxiv.org/abs/2107.14539" target="_blank">Arxiv</a> link / <a href="https://openaccess.thecvf.com/content/WACV2022/html/Sadekar_Shadow_Art_Revisited_A_Differentiable_Rendering_Based_Approach_WACV_2022_paper.html" target="_blank">CVF</a> link.

# Abstract

While recent learning based methods have been observed to be superior for several vision-related applications, their potential in generating artistic effects has not been explored much. One such interesting application is Shadow Art - a unique form of sculptural art where 2D shadows cast by a 3D sculpture produce artistic effects. In this work, we revisit shadow art using differentiable rendering based optimization frameworks to obtain the 3D sculpture from a set of shadow (binary) images and their corresponding projection information. Specifically, we discuss shape optimization through voxel as well as mesh-based differentiable renderers. Our choice of using differentiable rendering for generating shadow art sculptures can be attributed to its ability to learn the underlying 3D geometry solely from image data, thus reducing the dependence on 3D ground truth. The qualitative and quantitative results demonstrate the potential of the proposed framework in generating complex 3D sculptures that go beyond those seen in contemporary art pieces using just a set of shadow images as input. Further, we demonstrate the generation of 3D sculptures to cast shadows of faces, animated movie characters, and applicability of the framework to sketchbased 3D reconstruction of underlying shapes.


# Results

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/teaser.png" />
</div>

> Shadow art sculptures generated using differentiable rendering casting the shadows of (a) WACV acronym on one plane and fishes on the other resembling an aquarium of floating objects, (b) dropping Heart, Duck, and Mickey (all on the same plane), and (c) face sketches using half-toned images. (d) 3D reconstruction of a car from hand drawn sketches.


<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/sketches.png" />
</div>
>  A seemingly random voxel soup creates three distinct shadow images of (a) Albert Einstein, Nikola Tesla, and APJ Abdul Kalam, (b) Minions, and (c) Ironman


<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/objects.png" />
</div>

> 3D reconstruction of (a) flower vase, (b) pen-stand, and (c) coffee mug using the associated hand drawn sketches from three different views.


<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/printedart.jpg" />
</div>

> 3D Printed artistic sculptures generated using our proposed pipeline (quality upto the 3D printer's resolution).


# Paper

A preprint our work is currently available on arXiv. Click <a href="https://arxiv.org/abs/2107.14539" target="_blank">here</a> to access it.

# Citation

If you would like to cite us, kindly use the following BibTeX entry.

```
@InProceedings{Sadekar_2022_WACV,
    author    = {Sadekar, Kaustubh and Tiwari, Ashish and Raman, Shanmuganathan},
    title     = {Shadow Art Revisited: A Differentiable Rendering Based Approach},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {29-37}
}
```

# Acknowledgments
This research was supported by Science and Engineering Research Board (SERB) IMPacting Research INnovation and Technology (IMPRINT)-2 grant.

# Contact

Feel free to contact <a href="https://kaustubh-sadekar.github.io/" target="_blank">Kaustubh Sadekar</a> or <a href="https://www.linkedin.com/in/ashish-tiwari-82a392135/" target="_blank">Ashish Tiwari</a> for any further discussion about our work.

*Project page template inspired from [GradSLAM](https://gradslam.github.io/).*
