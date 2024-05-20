# ⚡Awesome WebNN [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

> A curated list of _awesome_ things related to the Web Neural Network (WebNN) API.

<a href="https://webnn.dev/"><img alt="WebNN logo" src="media/logo-webnn-white.svg" align="right" width="360" /></a>

[Web Neural Network (WebNN)](https://webnn.dev/) API is a new web standard that allows web apps and frameworks to accelerate deep neural networks with on-device hardware such as GPUs, CPUs, or purpose-built AI accelerators.

> Your contributions are always welcome! Please read the [contributing guidelines](CONTRIBUTING.md) to get started.

## WebNN Explained

- [WebNN Explainer](https://github.com/webmachinelearning/webnn/blob/main/explainer.md)

## Try out WebNN

- Chromium ([Chrome Canary](https://www.google.com/chrome/canary/), [Edge Canary](https://www.microsoftedgeinsider.com/download/canary), ...): Go to `about://flags#web-machine-learning-neural-network` and enable the "Enables WebNN API" flag
- WebNN Installation Guide from [W3C WebNN Samples](https://github.com/webmachinelearning/webnn-samples/#webnn-installation-guides), [WebNN Developer Preview Demos](https://xxxxxxxxx.github.io/webnn-developer-preview/install.html) or [Intel AI PC Development](https://www.intel.com/content/www/us/en/developer/topic-technology/ai-pc/webnn.html)

## Contents

- [Articles](#articles)
- [Blogs](#blogs)
- [Browser Support](#browser-support)
- [Community](#community)
- [Demos](#-demos)
  - [Demos on CPU or GPU](#demos-on-cpu-or-gpu)
  - [Demos on NPU](#demos-on-npu)
- [Frameworks](#frameworks)
- [Presentations](#presentations)
- [Samples](#-samples)
  - [Samples on CPU or GPU](#samples-on-cpu-or-gpu)
  - [Samples on NPU](#samples-on-npu)
- [Spec](#spec)
- [Testimonials](#testimonials)
- [Tutorials](#tutorials)
  - [Hugging Face Transformers](#hugging-face-transformers)
  - [Google JS ML Framework - WebNN](#google-js-ml-framework)
  - [MDN Docs](#mdn-docs)
  - [Microsoft ONNX Runtime Web](#microsoft-onnx-runtime-web)
  - [WebNN API](#webnn-api)
- [Videos](#videos)
- [Websites](#websites)

## Articles

- 2024.03 [Web-Apps smarter machen mit offlinefähigen KI-Modellen, WebGPU und WebNN](https://www.heise.de/blog/Web-Apps-smarter-machen-mit-offlinefaehigen-KI-Modellen-WebGPU-und-WebNN-7520733.html) - by Christian Liebel
- 2023.03 [Video Frame Processing on the Web – WebAssembly, WebGPU, WebGL, WebCodecs, WebNN, and WebTransport](https://webrtchacks.com/video-frame-processing-on-the-web-webassembly-webgpu-webgl-webcodecs-webnn-and-webtransport/) - by François Daoust

## Blogs

- 2024.04 [Updated Candidate Recommendation: WebNN API](https://www.w3.org/news/2024/updated-candidate-recommendation-web-neural-network-api/) - by Dominique Hazael-Massieux, Anssi Kostiainen

## Browser Support

- [Implementation Status of WebNN Operations](https://webmachinelearning.github.io/webnn-status/) ([data](https://github.com/webmachinelearning/webmachinelearning.github.io/blob/main/assets/json/webnn_status.json))
- [WebNN browser support overview](https://caniuse.com/?search=ml) ([data](https://github.com/mdn/browser-compat-data/pull/22569/files)) - CanIUse.com

## Community

- [Web Machine Learning Working Group](https://www.w3.org/groups/wg/webmachinelearning/) - W3C Community
- [Web Machine Learning Community Group](https://www.w3.org/groups/cg/webmachinelearning/) - W3C Community

## 💡 Demos

### Demos on CPU or GPU

- [Image Classification](https://xxxxxxxxx.github.io/webnn-developer-preview/demos/image-classification/) ([source](https://github.com/xxxxxxxxx/webnn-developer-preview/tree/demos-v1.01/demos/image-classification)) - EfficientNet Lite4, MobileNet V2, ResNet50
- [Segment Anything](https://xxxxxxxxx.github.io/webnn-developer-preview/demos/segment-anything/) ([source](https://github.com/xxxxxxxxx/webnn-developer-preview/tree/demos-v1.01/demos/segment-anything))
- [Stable Diffusion 1.5](https://xxxxxxxxx.github.io/webnn-developer-preview/demos/stable-diffusion-1.5/) ([source](https://github.com/xxxxxxxxx/webnn-developer-preview/tree/demos-v1.01/demos/stable-diffusion-1.5)) - Text Encoder, UNet, VAE, Safety Checker
- [Stable Diffusion Turbo](https://xxxxxxxxx.github.io/webnn-developer-preview/demos/sd-turbo/) ([source](https://github.com/xxxxxxxxx/webnn-developer-preview/tree/demos-v1.01/demos/sd-turbo)) - Text Encoder, UNet, VAE, Safety Checker
- [Whisper Base](https://xxxxxxxxx.github.io/webnn-developer-preview/demos/whisper-base/) ([source](https://github.com/xxxxxxxxx/webnn-developer-preview/tree/demos-v1.01/demos/whisper-base)) - audio, recording, and real time Whisper transcription

### Demos on NPU

- [Image Classification](https://xxxxxxxxx.github.io/webnn-developer-preview/demos/image-classification/) ([source](https://github.com/xxxxxxxxx/webnn-developer-preview/tree/demos-v1.01/demos/image-classification)) - EfficientNet Lite4, MobileNet v2, ResNet50 (coming soon)
- [Whisper Base](https://xxxxxxxxx.github.io/webnn-developer-preview/demos/whisper-base/) ([source](https://github.com/xxxxxxxxx/webnn-developer-preview/tree/demos-v1.01/demos/whisper-base)) - audio, recording, and real time Whisper transcription (coming soon)

## Frameworks

- [ONNX Runtime Web](https://onnxruntime.ai/)
- [Transformers.js](https://huggingface.co/docs/transformers.js)

## Presentations

- 2024.02 [WebNN: Die AI-Revolution im Browser?](https://basta.net/web-development/webbnn-api-ai-browser/) - by Christian Liebel
- 2023.10 [WebNN Implementation on DirectML](https://docs.google.com/presentation/d/1u9efG33BCIp0VdvpXXAu1yJlW10YvNNbnSklwJPBeCM/edit#slide=id.g24dab4effb5_0_0) - BlinkOn 18 - by Chai Chaoweeraprasit, Rafael Cintron, Ningxin Hu
- 2023.06 [WebNN Updates](https://ibelem.github.io/webnn-updates/) [PDF](https://www.w3.org/2023/06/china-web-forum/slides/zhang-min.pdf)

## 💡 Samples

### Samples on CPU or GPU

- [WebNN Samples](https://webmachinelearning.github.io/webnn-samples-intro/)([source](https://github.com/webmachinelearning/webnn-samples))
- [Code Editor](https://webmachinelearning.github.io/webnn-samples/code/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/code))
- [Face Recognition](https://webmachinelearning.github.io/webnn-samples/face_recognition/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/face_recognition)) - FaceNet, SSD MobileNet V2 Face
- [Facial Landmark Detection](https://webmachinelearning.github.io/webnn-samples/facial_landmark_detection/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/facial_landmark_detection)) - Face Landmark (SimpleCNN), SSD MobileNet V2 Face
- [Handwritten Digits Classification](https://webmachinelearning.github.io/webnn-samples/lenet/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/lenet)) - LeNet
- [Image Classification](https://webmachinelearning.github.io/webnn-samples/image_classification/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/image_classification)) - MobileNet v2, ResNet50 v2, SqueezeNet
- [Noise Suppression](https://webmachinelearning.github.io/webnn-samples/nsnet2/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/nsnet2)) - NSNet2
- [Noise Suppression](https://webmachinelearning.github.io/webnn-samples/rnnoise/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/rnnoise)) - RNNoise
- [Object Detection](https://webmachinelearning.github.io/webnn-samples/object_detection/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/object_detection)) - Tiny Yolo v2, SSD MobileNet v1
- [Semantic Segmentation](https://webmachinelearning.github.io/webnn-samples/semantic_segmentation/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/semantic_segmentation)) - DeepLab v3
- [Style Transfer](https://webmachinelearning.github.io/webnn-samples/style_transfer/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/style_transfer)) - Fast Style Transfer

### Samples on NPU

- [Image Classification](https://webmachinelearning.github.io/webnn-samples/image_classification/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/image_classification)) - EfficientNet Lite 4, MobileNet v2, ResNet 50 v1 (coming soon)
- [Object Detection](https://webmachinelearning.github.io/webnn-samples/object_detection/index.html) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/object_detection)) - SSD MobileNet v1 (coming soon)

## Spec

- [W3C Web Neural Network API](https://www.w3.org/TR/webnn/) - by Ningxin Hu, Chai Chaoweeraprasit, Dwayne Robinson

## Testimonials

## Tutorials

## Hugging Face Transformers

- To do

### Google JS ML Framework

- To do

### MDN Docs

- [WebNN](https://developer.mozilla.org/en-US/search?q=webnn) - To do

### Microsoft ONNX Runtime Web

- [WebNN Execution Provider](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)

### WebNN API

- [Introduction to Web Neural Network API (WebNN)](https://webmachinelearning.github.io/get-started/2024/05/16/introduction-to-web-neural-network-api.html) by Paul Cooper
- [Build Your First Graph with WebNN API](https://webmachinelearning.github.io/get-started/2021/03/15/build-your-first-graph-with-webnn-api.html)
- [Noise Suppression Net 2 (NSNet2)](https://webmachinelearning.github.io/get-started/2021/03/17/noise-suppression-net-v2.html)

## Videos

- 2023.10 [WebNN Implementation on DirectML](https://www.youtube.com/watch?v=FapumEVdrcg) - BlinkOn 18 - by Chai Chaoweeraprasit
- 2023.10 [AI @ W3C](https://www.youtube.com/watch?v=E0TbotgqAgw) by Dominique Hazael-Massieux
- 2021.12 [OpenCV Webinar 13: Chinese, Use WebNN to Optimize OpenCV.js DNN](https://www.youtube.com/watch?v=kQogwlhSsQ4) by Hanxi Guo
- 2021.11 [Introducing WebNN as a new backend for TensorflowJS](https://www.youtube.com/watch?v=v3LAY-Do25I) - by Shivay Lamba
- 2021.10 [WebNN Performance Comparison](https://www.youtube.com/watch?v=cHmWE5IHo9o&list=PLNhYw8KaLq2VOeJCyWZiEcmVYpXl5dw81) - W3C TPAC 2021 - by Wanming Lin

## Websites

- [Web Neural Network](https://webnn.dev/) - W3C Web Machine Learning WG
- [WebNN Developer Preview](https://xxxxxxxxx.github.io/webnn-developer-preview)
- [WebNN: Intel AI PC Development](https://www.intel.com/content/www/us/en/developer/topic-technology/ai-pc/webnn.html)

## Bug Reporting

- [Chromium](https://issues.chromium.org/issues?q=status:open%20componentid:1456206&s=created_time:desc)

## Source Code

- [Chromium](https://source.chromium.org/chromium/chromium/src/+/main:services/webnn/)

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0)