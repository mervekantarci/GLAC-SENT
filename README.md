# Visual Storytelling with GLAC-SENT

This repository modifies [GLACNet](https://github.com/tkim-snu/GLACNet)[[1]](#1) architecture to add textual 
cues using sentence encoder [Sentence-BERT](https://www.sbert.net)[[2]](#2).
Both text and image input are obtained from [Visual Storytelling Dataset (VIST) dataset](https://visionandlanguage.net/VIST/)[[3]](#3).

Please see the [report](https://github.com/mervekantarci/GLAC-SENT/blob/master/Report.pdf) for details of the model architecture, data preparation and achievements. 


## References
**IMPORTANT! The code is largely based on (and forked from) the great work at [GLACNet github repo](https://github.com/tkim-snu/GLACNet).**

<a id="1">[1]</a> 
Taehyeong Kim, Min-Oh Heo, Seonil Son, Kyoung-Wha Park, & Byoung-Tak Zhang (2018). GLAC Net: GLocal Attention Cascading Networks for Multi-image Cued Story Generation. CoRR, abs/1805.10973.


<a id="2">[2]</a> 
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.


<a id="3">[3]</a> 
Huang, T.H., Ferraro, F., Mostafazadeh, N., Misra, I., Devlin, J., Agrawal, A., Girshick, R., He, X., Kohli, P., Batra, D., & others (2016). Visual Storytelling. In 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2016).

