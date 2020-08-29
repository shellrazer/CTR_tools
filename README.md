# CTR_tools
This repository records some useful tools in machine learning of recommendation.
1. add bayesian smoothing class. I use moment method in sql calculation of conversion rate to solve 1) empty rate 2) ambiguous rate, eg: 1/1 has a higher rate than 9/10.
2. add common attention layers. I use additive attention in deepfm to give model more flexibility on regions. regions are different cluster of people. yes we do not train seperated model on different cluster of people. 
Implementation list  
|Name|Alignment score function|Citation|  
|---|---|---|  
|Additive|score(***s_t***, ***h_i***) = **v** tanh(**W**\[***s_t***;***h_i***\])|[Bahdanau 2015](https://arxiv.org/pdf/1409.0473.pdf)|  
|Dot-Product|score(***s_t***, ***h_i***) = ***s_t*** · ***h_i***|[Luong 2015](https://arxiv.org/pdf/1508.04025.pdf)|  
|Scaled Dot-Product|score(***s_t***, ***h_i***) = ***s_t*** · ***h_i*** / **d_k**|[Vaswani 2017](https://arxiv.org/abs/1706.03762)|  
