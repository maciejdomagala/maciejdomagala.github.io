---
layout: post
title:  "Recent rise of diffusion-based models"
date:   2022-06-06
categories: generative_models
comments: false
---

{% include mathjax.html %}
{% include styles.html %}

## Introduction

Every fan of generative modeling is living an absolute dream for the last year and the half (at least!). Smart people from big companies decided to spoil us with several new developments/papers on the text-to-image generation, each one arguably better than previous. As an effect, we have been observing a social media surge of spectacular, purely AI generated images, such as this golden retriever answering tough questions during his political campaign or a brain riding a rocketship to the moon.

![](/assets/images/combine_images.png)
*Sources: [https://openai.com/dall-e-2/](https://openai.com/dall-e-2/) and [https://imagen.research.google/](https://imagen.research.google/)*

In this post we will sum up the very recent history of solving the text-to-image generation problem and explain the latest developments regarding diffusion models, which are playing a huge role in the new, state-of-the-art architectures.

![Short timeline of image generation and text-to-image solutions.](/assets/images/Screenshot_2022-05-31_at_13.31.46.png)

*Short timeline of image generation and text-to-image solutions. Source: author*

## DALL·E

In 2020 OpenAl team [Brown et al., 2020] published GPT-3 model - a multi-modal does-it-all huge language model, capable of machine translation, text generation, semantic analysis etc. Model quickly became regarded as the state-of-the-art for language modeling solutions and DALL·E [Ramesh et al. 2021] can be viewed as a natural expansion of the transformer capabilities. 

### Autoregressive approach

Authors proposed an elegant two stage approach:

- train a discrete VAE model for image compression into image tokens,
- concatenate encoded text snippet with the image tokens and train the autoregressive transformer to learn the joint distribution over text and images.

The final version was trained on 250 million text-image pairs obtained from the internet.

### CLIP

During inference, the model is able to output a whole batch of generated images. But how can we estimate which images are *best*? Simultaneously with the publication of DALL·E, OpenAI team presented a solution for image and text linking called CLIP [Radford et al., 2021]. In a nutshell, CLIP offers a reliable way of pairing a text snippet with its image representation. Putting aside all of the technical aspects, the idea of training this type of model is fairly simple - take the text snippet and encode it, take an image and encode it. Do that for a lot of examples (400 millions of (image, text) pairs) and train the model in *contrastive* fashion. 

![Visualisation of CLIP contrastive pre-training, source: [https://arxiv.org/pdf/2103.00020.pdf](https://arxiv.org/pdf/2103.00020.pdf)](/assets/images/Untitled.png)

*Visualisation of CLIP contrastive pre-training, source: [https://arxiv.org/pdf/2103.00020.pdf](https://arxiv.org/pdf/2103.00020.pdf)*

Having this kind of *mapping* allows us to estimate which of the generated images is a best match considering the text input. Anyone who would like to see the power of CLIP - feel free to check out my previous post on combining CLIP and evolutionary algorithms to generate images.

DALL·E caught a lot of attention from people inside and outside AI world, it gained lots of publicity and stirred a lot of conversation. Even though - it is only a honorable mention here, as the fashion quite quickly shifts towards novel ideas.

## All you need is diffusion

[Sohl-Dickstein et al., 2015] proposed a fresh idea on the subject of image generation - diffusion models.

![Generative models, source: [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)](/assets/images/Screenshot_2022-05-31_at_11.12.14.png)

*Generative models, source: [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)*

Idea is quite straightforward, inspired by non-equilibrium thermodynamics, although underneath it is packed with some interesting mathematical concepts. Here, we are also seeing a concept of encoder-decoder type of structure, but the underlying idea is a bit different than what we can observe in traditional variational autoencoders. To understand the basics this model, we need to describe forward and reverse diffusion processes.

### Forward image diffusion

Altogether, this process can be described as gradually applying Gaussian noise to the image until it becomes entirely unrecognisable. This process is fixed in stochastic sense - the procedure of noise application can be formulated as the Markov chain of sequential diffusion steps. To untangle the difficult wording a little bit, we can neatly describe it with a few formulas. Assume that images have a certain starting distribution $$q(\bf{x}_{0})$$. We can sample just one image from this distribution - $$\bf{x}_{0}$$. We want to perform a chain of diffusion steps $$\bf{x}_{0} \to \bf{x}_{1} \to ... \to \bf{x}_{\it{T}}$$ , each step disintegrating the image more and more. 

![Visualisation of gaussian noising procedure](/assets/images/Untitled%201.png)

*Visualisation of gaussian noising procedure. Source: author*

How exactly is the noise applied? It is formally defined by a *noising schedule $$\{\beta_{t}\}^{T}_{t=1}$$,* where for every $$t = 1,...,T$$ we have $$\beta_{t} \in (0,1)$$. With such schedule we can formally define the process as

$$
      q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\sqrt{1-\beta_{t}} \mathbf{x}_{t-1}, \beta_{t} \mathbf{I}\right).
$$

That is about it on the forward process, there are just two more things worth mentioning:

- As the number of noising steps is going up $$(T \to \infty)$$, the final distribution $$q(\mathbf{x}_{T})$$ approaches a very handy isotropic Gaussian distribution. That makes any future sampling from *noised* distribution efficient and easy.
- Noising with Gaussian kernel provides another benefit - there is no need to walk step-by-step through noising process to achieve any intermediate latent state. We can sample them directly thanks to reparametrization
    
    $$
    q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)=\mathcal{N}\left(\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0},\left(1-\bar{\alpha}_{t}\right) \mathbf{I}\right) = \sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \cdot \epsilon,
    $$
    
    where $$\alpha_{t} := 1-\beta_{t}$$, $$\bar{\alpha}_{t} := \prod_{k=0}^{t}\alpha_{k}$$ and $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$. Here $$\epsilon$$ represents Gaussian noise - this formulation will be essential for the model training.
    

### Reverse image diffusion

We have a nicely defined process. Someone might ask - so what? Why can’t we just define a reverse process $$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$$ and trace back from the noise to the image? First of all, that would fail conceptually, as we want to have a neural network that *learns* how to deal with a problem - we shouldn’t provide it with a clear solution. And second of all, we cannot quite do that. It would require a marginalization over the entire data distribution. To get back to the starting distribution $$q(\bf{x}_{0})$$ from the noised sample we would have to marginalize over all of the ways we could arise at $$\mathbf{x}_{0}$$ from the noise, including all of the latent states inbetween - $$\int q(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}$$ - which is intractable. So, if we cannot calculate it, surely we can… approximate it!

The core idea is to develop a reliable solution - in the form of learnable network - that successfully approximates the reverse diffusion process. First way is to achieve that by estimating mean and covariance for denoising steps

$$
p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)=\mathcal{N}(\mu_{\theta}(\mathbf{x}_{t}, t), \Sigma_{\theta}(\mathbf{x}_{t}, t) ).
$$

In a practical sense, $$\mu_{\theta}(\mathbf{x}_{t}, t)$$ can be estimated via neural network and $$\Sigma_{\theta}(\mathbf{x}_{t}, t)$$ can be fixed to a certain constant related with the noising schedule, such as $$\beta_{t}\mathbf{I}$$.

![Forward and reverse diffusion processes, source: [Angus Turner, 2021]](/assets/images/Untitled%202.png)
*Forward and reverse diffusion processes, source: [Angus Turner, 2021]*

Estimating $$\mu_{\theta}(\mathbf{x}_{t}, t)$$ this way is possible, and was done, but [Ho et al., 2020] came up with a different way of training - a neural network $$\epsilon_{\theta}(\mathbf{x}_{t}, t)$$ can be trained to predict the noise $$\epsilon$$ from the earlier formulation of $$q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)$$.

As in [Ho et al., 2020], the training process consists of steps:

1. Sample image $$\mathbf{x}_{0}\sim q(\bf{x}_{0})$$,
2. Choose a certain step in diffusion process $$t \sim U(\{1,2,...,T\})$$,
3. Apply the noising $$\epsilon \sim \mathcal{N}(0,\mathbf{I})$$,
4. Try to estimate the noise $$\epsilon_{\theta}(\mathbf{x}_{t}, t)= \epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \cdot \epsilon, t)$$,
5. Learn the network by gradient descent on loss $$\nabla_{\theta}  \|\epsilon - \epsilon_{\theta}(\mathbf{x}_{t}, t)\|^{2}$$.

In general, loss can be nicely presented as

$$
L_{\text{diffusion}}=\mathbb{E}_{t, \mathbf{x}_{0}, \epsilon}\left[\left\|\epsilon-\epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)\right\|^{2}\right],
$$

where $$t, \mathbf{x}_0$$ and $$\epsilon$$ are described as in the steps above.

All of the formulations, reparametrizations and derivations can be a bit math extensive, but there are already some great resources available for anyone that wants to get a deeper understanding on the subject. Most notably, [Lillian Weng, 2021], [Angus Turner, 2021] and [Ayan Das, 2021] went through some deep derivations while mantaining understandable tone - highly recommend to check these posts.

### Guiding the diffusion

Above part itself explains how we can perceive diffusion model as generative. Once the model $$\epsilon_{\theta}(\mathbf{x}_{t}, t)$$ is trained we can use it to run the noise $$\mathbf{x}_{t}$$ back to $$\mathbf{x}_{0}$$. Given that it is easy to sample the noise from isotropic Gaussian distribution, we can obtain limitless image variations. We can also guide the image generation, by feeding additional information to the network during the training process. Assuming that the images are labelled, the information about class $$y$$ can be fed into a class-conditional diffusion model $$\epsilon_{\theta}(\mathbf{x}_{t}, t \mid y)$$.

One way of introducing the guidance in the training process is to train a separate model, which acts as a classifier of noisy images. At each step of denoising the classifier checks whether the image is denoised *in the right direction* and contributes with its own gradient of loss function into the overall loss of diffusion model.

[Ho & Salimans, 2021] proposed an idea on how to feed the class information into the model without the need of training additional classifier. During the training the model $$\epsilon_{\theta}(\mathbf{x}_{t}, t \mid y)$$ is sometimes (with fixed probability) not being shown the actual class $$y$$. Instead, the class label is replaced with the null label $$\emptyset$$. So it learns to perform diffusion with and without the guidance. For inference, the model is performing two predictions, once given the class label $$\epsilon_{\theta}(\mathbf{x}_{t}, t \mid y)$$ and once not $$\epsilon_{\theta}(\mathbf{x}_{t}, t \mid \emptyset)$$. The final prediction of the model is moved away from $$\epsilon_{\theta}(\mathbf{x}_{t}, t \mid \emptyset)$$ and towards $$\epsilon_{\theta}(\mathbf{x}_{t}, t \mid y)$$ by scaling with *guidance scale $$s \geqslant 1$$.*

$$
\hat{\epsilon}_{\theta}\left(\mathbf{x}_{t}, t \mid y\right)=\epsilon_{\theta}\left(\mathbf{x}_{t}, t \mid \emptyset\right)+s \cdot\left(\epsilon_{\theta}\left(\mathbf{x}_{t}, t \mid y\right)-\epsilon_{\theta}\left(\mathbf{x}_{t}, t \mid \emptyset\right)\right)
$$

This kind of classifier-free guidance uses only the main model’s comprehension - additional classifier is not needed - which yields better results according to [Nichol et al. 2021].

## GLIDE

Even though the paper describing GLIDE [Nichol et al., 2021] architecture received the least publicity out of all of the publications discussed in this post, it arguably presents the most novel and interesting ideas. It nicely combines all of the concepts presented in the previous chapter. We already know how diffusion models are working and that we can use them to generate images. Two questions we would like to answer now are:

- How can we use the textual information to guide the diffusion model?
- How can we make sure that the quality of the model is good enough?

### Architecture choice

Three main components can be distinguished from the architecture: 

1. UNet based model responsible for the visual part of the diffusion learning,
2. Transformer based model responsible for creating a text embedding from text snippet,
3. Upsampling diffusion model used for enhancing output image resolution.

First two are working together in order to create a text-guided image output, the last one is used for enlarging the image while preserving the quality.

The core of the model is the well-known UNet architecture, used for the diffusion in [Dhariwal & Nichol, 2021]. The model, just like in early versions, stacks residual layers with downsampling and upsampling convolutions. It also consists of attention layers which are crucial for simultaneous text processing. Model proposed by authors has around 2.3 billion parameters and was trained on the same dataset as DALL·E.

The text used for guidance is encoded in tokens and fed into Transformer model. The model used in GLIDE had roughly 1.2 billion parameters and was built from 24 residual blocks of width 2048. Output of the transformer has two purposes:

- final embedding token is used as class embedding $$y$$ in $$\epsilon_{\theta}(\mathbf{x}_{t}, t \mid y)$$,
- final layer of token embeddings is added to **every** attention layer of the model.

It is clearly visible that a lot of focus was put into making sure that the model receives enough text-related context in order to generate accurate images. Model is conditioned on the text snippet embedding, encoded text is concatenated with the attention context and during training the classifier-free guidance is used.

As for the final component, authors used the diffusion model to go from low resolution to high resolution image. For that, they used a ImageNet upsampler.

![GLIDE interpretation of ‘a corgi in a field’, source: [https://arxiv.org/pdf/2112.10741.pdf](https://arxiv.org/pdf/2112.10741.pdf)](/assets/images/Untitled%203.png)

*GLIDE interpretation of ‘a corgi in a field’, source: [https://arxiv.org/pdf/2112.10741.pdf](https://arxiv.org/pdf/2112.10741.pdf)*

GLIDE is incorporating a few notable achievements developed in the recent years and sheds a new light into the text-guided image generation concept. Given that DALL·E model was based on different structures, it is fair to say that publication of GLIDE starts the diffusion-based text-to-image generation era.

## DALL·E 2

OpenAI’s team doesn’t seem to get much rest, as in April they took the Internet by the storm with DALL·E 2 [Ramesh et al., 2022]. It takes a bit from both predecessors: it heavily relies on CLIP [Radford et al., 2021] but the large part of solution revolves around GLIDE [Nichol et al., 2021] architecture. DALL·E 2 has two main underlying components called *prior* and *decoder,* which stacked together are able to produce image output. The whole mechanism was named *unCLIP,* which can already spoil a bit of mystery on what exactly is going on under the hood.

![Visualization of DALL·E 2 two-stage mechanism. Source: [https://arxiv.org/pdf/2204.06125.pdf](https://arxiv.org/pdf/2204.06125.pdf)](/assets/images/Untitled%204.png)

*Visualization of DALL·E 2 two-stage mechanism. Source: [https://arxiv.org/pdf/2204.06125.pdf](https://arxiv.org/pdf/2204.06125.pdf)*

### The prior

First stage is meant to convert the caption - text snippet such as *a “corgi playing a flame throwing trumpet” -* into a text embedding. We obtain it using a frozen CLIP model. 

After text embedding is obtained comes the fun part - we want now to obtain an image embedding, similar to the one which is obtained via CLIP model. We want it to encapsulate all important information from the text embedding, as it will be used for image generation through diffusion. Well, isn’t that exactly what CLIP is for? If we want to find a respective image embedding for our input phrase, we can just look what is close to our text embedding in the CLIP encoded space. One of the DALL·E 2 authors [Aditya Ramesh, 2022] posted a nice explanation of why that solution fails and why the prior is needed - “An infinite number of images could be consistent with a given caption, so the outputs of the two encoders will not perfectly coincide. Hence, a separate prior model is needed to “translate” the text embedding into an image embedding that could plausibly match it”. 

On top of that, authors checked the importance of the prior in the network empirically. Passing both the image embedding produced by the prior and the text vastly outperforms generation using only caption or caption with CLIP text embedding.

![Samples generated conditioned on: caption, text embedding and image embedding. Source: [https://arxiv.org/pdf/2204.06125.pdf](https://arxiv.org/pdf/2204.06125.pdf)](/assets/images/Untitled%205.png)

*Samples generated conditioned on: caption, text embedding and image embedding. Source: [https://arxiv.org/pdf/2204.06125.pdf](https://arxiv.org/pdf/2204.06125.pdf)*

Authors tested two model classes for the prior: autoregressive model and diffusion model. This post will cover only the diffusion prior, as it was deemed better performing than autoregressive, especially from computational point of view. For the training of the prior a decoder-only Transformer model was chosen. It was trained by using a sequence of several inputs:

- encoded text,
- CLIP text embedding,
- embedding for the diffusion timestep,
- noised image embedding,

with the goal of outputting an unnoised image embedding $$z_{i}$$. On the contrary to the way of training proposed by [Ho et al., 2020] covered in previous sections, predicting the unnoised image embedding directly instead of predicting the noise was a better fit. So, remembering the previous formula for diffusion loss in a guided model

$$
L_{\text{diffusion}}=\mathbb{E}_{t, \mathbf{x}_{0}, \epsilon}\left[\left\|\epsilon-\epsilon_{\theta}\left(\mathbf{x}_{t}, t\mid y\right)\right\|^{2}\right],
$$

we can present the prior diffusion loss as

$$
L_{\text{prior:diffusion}}=\mathbb{E}_{t}\left[\left\|z_{i}-f_{\theta}\left({z}_{i}^{t}, t \mid y\right)\right\|^{2}\right],
$$

where $$f_{\theta}$$ stands for the prior model, $${z}_{i}^{t}$$ is the noised image embedding, $$t$$ is the timestamp and $$y$$ is the caption used for guidance. 

### The decoder

We covered the prior part of the unCLIP, which was meant to produce a model that is able to encapsulate all of the important information from the text into a CLIP-like image embedding. Now we want to use that image embedding to generate an actual visual output. Now is when the name *unCLIP* unfolds itself - we are walking back from the image embedding to the image, in reverse to what happens when CLIP image encoder is trained.

As the saying goes: “After one diffusion model it is time for another diffusion model!”. And this one we already know - it is GLIDE, although slightly modified. Only *slightly*, since single major change is adding the additional CLIP image embedding (produced by the prior) to the vanilla GLIDE text encoder. After all, this is exactly what the prior was trained for - to provide information for the decoder. Just as in regular GLIDE, guidance is used. To improve it, CLIP embeddings are set to $$\emptyset$$ in 10% of cases and text captions $$y$$ in 50% of cases.

Another thing that did not change is the idea of upsampling after the image generation. The output is tossed into additional diffusion-based models. This time two umsampling models are used (instead of one in original GLIDE), one taking the image from 64x64 to 256x256 and the other further enhancing resolution up to 1024x1024.

## Imagen

Google Brain team decided not to be late to the party, as not even two months after DALL·E 2 publication they presented their work - Imagen [Saharia et al., 2022].

![Overview of Imagen architecture. Source: [https://arxiv.org/pdf/2205.11487.pdf](https://arxiv.org/pdf/2205.11487.pdf)](/assets/images/Untitled%206.png)

*Overview of Imagen architecture. Source: [https://arxiv.org/pdf/2205.11487.pdf](https://arxiv.org/pdf/2205.11487.pdf)*

Imagen architecture seems to be oddly simple in its structure. A pretrained textual model is used to create the embeddings that are diffused into an image. Next, the resolution is increased via super-resolution diffusion models - the steps we already know from DALL·E 2. A lot of novelties are scattered in different bits of the architecture - few in the model itself and several in the training process. Together, they offer a slight upgrade when comparing with other solutions. Given a large portion of knowledge already served, we can explain this model via differences with previously described models:

**Use a pretrained transformer instead of training it from the scratch.**
This is stated as the core improvement compared to OpenAI’s work. For everything regarding text embeddings, GLIDE authors used a new, specifically trained transformer model. Imagen authors used a pretrained, frozen T5-XXL model [Roberts et al., 2020]. The idea is that this model has a vastly more context regarding language processing than a model trained only on the image captions and hence is able to produce more valuable embeddings without the need to additionally fine-tune it.

**Make the underlying neural network more efficient.**
An upgraded version of neural network called *Efficient U-net* was used as the backbone of super-resolution diffusion models. It is said to be more memory efficient and simpler than previous version, also it converges faster. The changes were introduced mainly in residual blocks and via additional scaling of the values inside of the network. For anyone enjoying going deep into details - changes are well documented in [Saharia et al., 2022].

**Use *conditioning augmentation* to enhance image fidelity**
Since the solution can be viewed as a sequence of diffusion models, there is an argument to be made about enhancements in the areas where the models are linking. [Ho et al., 2021] presented a solution called *conditioning augmentation.* In simple words, it is equivalent to applying various data augmentation techniques, such as Gaussian blur, to the low resolution image before it is fed into the super-resolution models.

There are a few other resources deemed crucial for the low FID score and high image fidelity (such as *dynamic thresholding*) - these are explained in details in the source paper [Saharia et al., 2022]. The core of the approach is already covered in previous chapters. 

![Some of Imagen generations with captions. Source: [https://arxiv.org/pdf/2205.11487.pdf](https://arxiv.org/pdf/2205.11487.pdf)](/assets/images/Untitled%207.png)

*Some of Imagen generations with captions. Source: [https://arxiv.org/pdf/2205.11487.pdf](https://arxiv.org/pdf/2205.11487.pdf)*

### Is it *the best* yet?

As of writing this text, Google’s Imagen is considered to be state-of-the-art as far as text-to-image generation is concerned. But why exactly is that? How can we evaluate the models and compare them against each other?

Authors of Imagen opted for two ways of evaluation. One is considered to be a current standard for text-to-image modelling, which is establishing a Fréchet inception distance score on a COCO validation dataset. Authors report (*unsurprisingly*) that Imagen shows a state-of-the-art performance, its *zero-shot* FID outperforming all other models, even these specifically trained on COCO.

![source: [https://arxiv.org/pdf/2205.11487.pdf](https://arxiv.org/pdf/2205.11487.pdf)](/assets/images/Untitled%208.png)

*Comparison of several models. Source: [https://arxiv.org/pdf/2205.11487.pdf](https://arxiv.org/pdf/2205.11487.pdf)*

Far more intruiging way of evaluation is a brand new proposal from the authors called DrawBench - *a comprehensive and challenging set of prompts that support the evaluation and comparison of text-to-image models (source).* It consists of 200 prompts falling into 11 categories, collected from e.g. DALL·E or Reddit. List of the prompts with categories can be found in [DrawBench prompts, 2022] Evaluation was performed by 275 *unbiased* (sic!) raters, 25 for each category. Each rater was shown two *non-cherry picked* and *random* sets of images generated by two different models (e.g. Imagen and DALL·E 2) and had to respond to two questions:

1. Which set of images is of higher quality?
2. Which set of images better represents the text caption?

These two questions are meant to address two most important characteristics of a good text-to-image model: quality of produced images (fidelity) and how well it reflects the input text prompt (alignment). Each rater had three choices - to claim that one of the models is performing better or to call it a tie. Once again - there can be only one winner. Interestingly, GLIDE model seems to be performing a bit better than DALL·E 2, at least on this curated dataset.

![Source: https://arxiv.org/pdf/2205.11487.pdf](/assets/images/Untitled%209.png)

*Imagen vs other models. Source: https://arxiv.org/pdf/2205.11487.pdf*

As expected, a large portion of the publication is devoted to the comparison between the images produced by Imagen and GLIDE/DALL·E - more can be found in Appendix E of [Saharia et al., 2022].

## The fun is far from over

As usual, with the new architecture gaining recognition there is a large surge of interesting publications and solutions emerging from the void. The pace of developments makes it nearly impossible to track every interesting publication. There are also a lot of interesting characteristics  of the models to discover other than raw generative power, such as image inpainting, style transfer and image editing.

Apart from the understandable excitement over a new era of generative models, there are some shortcomings embedded into the diffusion process structure, such as slow sampling speed compared to previous models [Vahdat and Kreis, 2022].

![Source: [https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/](https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/)](/assets/images/Untitled%2010.png)

*Source: [https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/](https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/)*

For anyone who likes to get deep into the bits of implementation, I highly recommend going through Phil Wang’s (@lucidrains on github) repositories [Phil Wang’s repositories], which is a collaborative effort from many people to recreate the unpublished models in PyTorch.

For anyone who would like to admire some more examples of DALL·E 2 generative power, I recommend checking newly created subreddit with DALL·E 2 creations in [DALL·E 2 subreddit]. It is moderated by people with OpenAI’s Lab access - feel free to join the waitlist [OpenAI’s waitilist] and have the opportunity to play with models yourself.

## References

### Publications

- [Sohl-Dickstein et al., 2015] Deep Unsupervised Learning using
Nonequilibrium Thermodynamics, [https://arxiv.org/pdf/1503.03585.pdf](https://arxiv.org/pdf/1503.03585.pdf)
- [Ho et al., 2020] Denoising diffusion probabilistic models, [https://arxiv.org/pdf/2006.11239.pdf](https://arxiv.org/pdf/2006.11239.pdf)
- [Brown et al., 2020] Language Models are Few-Shot Learners, [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- [Roberts et al., 2020] How Much Knowledge Can You Pack Into the Parameters of a Language Model?, [https://arxiv.org/pdf/2002.08910.pdf](https://arxiv.org/pdf/2002.08910.pdf)
- [Ho & Salimans, 2021] Classifier-Free Diffusion Guidance, [https://openreview.net/pdf?id=qw8AKxfYbI](https://openreview.net/pdf?id=qw8AKxfYbI)
- [Nichol et al., 2021] GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models, [https://arxiv.org/abs/2112.10741](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbC1Ya3NvVzV0NG5xcEtfeHE2cloxUG16YWhuQXxBQ3Jtc0trZGttQm9tenh3UnAwSGo0U3JoeFZpT0E5cVFFX1JjQURIb2NpeV9SelB2dFRDbDJWRXE4R1EzM0QxUjRUVmp6WDhEOGh2OVEtTjlRTUw0N04tU1hnNzNINFhZSzduZVNSejBUYjAxeW8ySTFQcXg5Yw&q=https%3A%2F%2Farxiv.org%2Fabs%2F2112.10741&v=fbLgFrlTnGU)
- [Ramesh et al. 2021] Zero-Shot Text-to-Image Generation, [https://arxiv.org/pdf/2102.12092.pdf](https://arxiv.org/pdf/2102.12092.pdf)
- [Dhariwal & Nichol, 2021] Diffusion Models Beat GANs on Image Synthesis, [https://arxiv.org/pdf/2105.05233.pdf](https://arxiv.org/pdf/2105.05233.pdf)
- [Radford et al., 2021] Learning Transferable Visual Models From Natural Language Supervision, [https://arxiv.org/pdf/2103.00020.pdf](https://arxiv.org/pdf/2103.00020.pdf)
- [Ho et al., 2021], Cascaded Diffusion Models for High Fidelity Image Generation, [https://arxiv.org/pdf/2106.15282.pdf](https://arxiv.org/pdf/2106.15282.pdf)
- [Ramesh et al., 2022] Hierarchical Text-Conditional Image Generation with CLIP Latents, [https://arxiv.org/pdf/2204.06125.pdf](https://arxiv.org/pdf/2204.06125.pdf)
- [Saharia et al., 2022] Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding, [https://arxiv.org/pdf/2205.11487.pdf](https://arxiv.org/pdf/2205.11487.pdf)

### Blog posts

- [Angus Turner, 2021] “Diffusion Models as a kind of VAE”, [https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html)
- [Lillian Weng, 2021] “What are Diffusion Models?”, [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Ayan Das, 2021] “An introduction to Diffusion Probabilistic Models”, [https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)
- [Aditya Ramesh, 2022] How DALL·E 2 works, [http://adityaramesh.com/posts/dalle2/dalle2.html](http://adityaramesh.com/posts/dalle2/dalle2.html)
- [Vahdat and Kreis, 2022], Improving Diffusion Models as an Alternative To GANs, Part 1, [https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/](https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/)

### Misc

- [DrawBench prompts], [DrawBench](https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/htmlview?pru=AAABgRqAJJQ*agF3cOZ-eQVuWLxxWEwiWQ#gid=0)
- [DALL·E 2 subreddit], [https://www.reddit.com/r/dalle2/](https://www.reddit.com/r/dalle2/)
- [OpenAI’s waitilist], [https://labs.openai.com/waitlist](https://labs.openai.com/waitlist)
- [Phil Wang’s repositories], [https://github.com/lucidrains?tab=repositories](https://github.com/lucidrains?tab=repositories)