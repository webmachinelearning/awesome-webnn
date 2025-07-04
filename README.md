# ⚡Awesome WebNN [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

> A curated list of _awesome_ things related to the Web Neural Network (WebNN) API.

<img alt="WebNN logo" src="media/logo-webnn-white.svg" align="right" width="360" />

[Web Neural Network (WebNN)](https://webnn.dev/) API is a new web standard that allows web apps and frameworks to accelerate deep neural networks with on-device hardware such as GPUs, CPUs, or purpose-built AI accelerators.

> Your contributions are always welcome! Please read the [contributing guidelines](CONTRIBUTING.md) to get started.

## WebNN Explained

- [WebNN Explainer](https://github.com/webmachinelearning/webnn/blob/main/explainer.md)

## Try out WebNN

- Go to `about://flags#web-machine-learning-neural-network` and enable the "Enables WebNN API" flag with [Google Chrome Dev](https://www.google.com/chrome/dev/) or [Microsoft Edge Dev](https://www.microsoft.com/edge/download/insider)
- WebNN Installation Guide from [W3C WebNN Samples](https://github.com/webmachinelearning/webnn-samples/#webnn-installation-guides), [WebNN Developer Preview Demos](https://microsoft.github.io/webnn-developer-preview/install.html) or [Intel AI PC Development](https://www.intel.com/content/www/us/en/developer/topic-technology/ai-pc/webnn.html)

## Contents

- [Articles](#articles)
- [Blogs](#blogs)
- [Browser Support](#browser-support)
- [Community](#community)
- [Demos](#demos)
  - [Demos on CPU or GPU](#demos-on-cpu-or-gpu)
  - [Demos on NPU](#demos-on-npu)
  - [Other Demos](#other-demos)
- [Frameworks](#frameworks)
- [Presentations](#presentations)
- [Samples](#samples)
  - [Samples on CPU or GPU](#samples-on-cpu-or-gpu)
  - [Samples on NPU](#samples-on-npu)
- [Spec](#spec)
- [Testimonials](#testimonials)
- [Tutorials](#tutorials)
  - [ONNX Runtime Web](#onnx-runtime-web)
  - [WebNN API](#webnn-api)
- [Videos](#videos)
- [Websites](#websites)

## Articles

- 2025.06 [AI in the Runtime: Why WebNN Changes Everything](https://medium.com/rethinking-the-client-a-new-era-of-modular/ai-in-the-runtime-why-webnn-changes-everything-70494a3cf524) by Enrico Piovesan
- 2024.12 [How the Web Neural Network API Brings AI Power to Your Browser](https://javascript.plainenglish.io/how-the-web-neural-network-api-brings-ai-power-to-your-browser-8bef925954f8) by Ferid Brković
- 2024.07 [Web AI Monthly #20: Microsoft / Intel launch WebNN / WebGPU playgrounds](https://www.linkedin.com/pulse/web-ai-monthly-20-collaboration-hugging-face-global-talks-jason-mayes-lwgdc/) by Jason Mayes
- 2024.03 [Web-Apps smarter machen mit offlinefähigen KI-Modellen, WebGPU und WebNN](https://www.heise.de/blog/Web-Apps-smarter-machen-mit-offlinefaehigen-KI-Modellen-WebGPU-und-WebNN-7520733.html) - by Christian Liebel
- 2023.03 [Video Frame Processing on the Web – WebAssembly, WebGPU, WebGL, WebCodecs, WebNN, and WebTransport](https://webrtchacks.com/video-frame-processing-on-the-web-webassembly-webgpu-webgl-webcodecs-webnn-and-webtransport/) - by François Daoust

## Blogs

- 2024.12 [WebNN: Bridging the Gap Between AI and the Web](https://www.nsdbytes.com/web-neural-network-api-webnn-bridging-the-gap-between-ai-and-the-web/)
- 2024.06 [WebNN: Bringing AI Inference to the Browser](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/webnn-bringing-ai-inference-to-the-browser/ba-p/4175003) by Sapna Giddegowda
- 2024.05 [Intel: Announcing WebNN Developer Preview for the AI PC](https://www.intel.com/content/www/us/en/developer/articles/news/announcing-webnn-developer-preview-for-the-ai-pc.html) - by Qi Zhang
- 2024.05 [Microsoft: Introducing the WebNN Developer Preview with DirectML](https://blogs.windows.com/windowsdeveloper/2024/05/24/introducing-the-webnn-developer-preview-with-directml/) - by Adele Parsons, Dwayne Robinson
- 2024.05 [Microsoft: WebNN Developer Preview through DirectML announced at Build 2024](https://blogs.windows.com/windowsdeveloper/2024/05/21/unlock-a-new-era-of-innovation-with-windows-copilot-runtime-and-copilot-pcs/) - by Paval Davuluri
- 2024.05 [NVIDIA: WebNN accelerated with NVIDIA RTX via DirectML announced at Build 2024](https://blogs.nvidia.com/blog/rtx-advanced-ai-windows-pc-build/) by Jesse Clayton
- 2024.04 [W3C: Updated WebNN API Candidate Recommendation](https://www.w3.org/news/2024/updated-candidate-recommendation-web-neural-network-api/) - by Dominique Hazael-Massieux, Anssi Kostiainen

## Browser Support

- [Implementation Status of WebNN Operations](https://webmachinelearning.github.io/webnn-status/) ([data](https://github.com/webmachinelearning/webmachinelearning.github.io/blob/main/assets/json/webnn_status.json))
- [WebNN browser support overview](https://caniuse.com/?search=webnn) ([data](https://github.com/mdn/browser-compat-data/pull/22569/files)) - CanIUse.com

## Community

- [Web Machine Learning Working Group](https://www.w3.org/groups/wg/webmachinelearning/) - W3C Community
- [Web Machine Learning Community Group](https://www.w3.org/groups/cg/webmachinelearning/) - W3C Community

## Demos

### Demos on CPU or GPU

- [Image Classification](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/) ([source](https://github.com/microsoft/webnn-developer-preview/tree/main/demos/image-classification)) - EfficientNet Lite4, MobileNet V2, ResNet50
- [Segment Anything](https://microsoft.github.io/webnn-developer-preview/demos/segment-anything/) ([source](https://github.com/microsoft/webnn-developer-preview/tree/main/demos/segment-anything))
- [Stable Diffusion 1.5](https://microsoft.github.io/webnn-developer-preview/demos/stable-diffusion-1.5/) ([source](https://github.com/microsoft/webnn-developer-preview/tree/main/demos/stable-diffusion-1.5)) - Text Encoder, UNet, VAE, Safety Checker
- [Stable Diffusion Turbo](https://microsoft.github.io/webnn-developer-preview/demos/sd-turbo/) ([source](https://github.com/microsoft/webnn-developer-preview/tree/main/demos/sd-turbo)) - Text Encoder, UNet, VAE, Safety Checker
- [Whisper Base](https://microsoft.github.io/webnn-developer-preview/demos/whisper-base/) ([source](https://github.com/microsoft/webnn-developer-preview/tree/main/demos/whisper-base)) - Audio, recording, and real time Whisper transcription

### Demos on NPU

- [Image Classification](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/) ([source](https://github.com/microsoft/webnn-developer-preview/tree/main/demos/image-classification)) - EfficientNet Lite4, MobileNet v2, ResNet50 (coming soon)
- [Whisper Base](https://microsoft.github.io/webnn-developer-preview/demos/whisper-base/) ([source](https://github.com/microsoft/webnn-developer-preview/tree/main/demos/whisper-base)) - Audio, recording, and real time Whisper transcription (coming soon)

### Other Demos

- [Angular AI (Analysis, Vision, Generation)](https://ng-ai-zeta.vercel.app/) ([source](https://github.com/webmaxru/ng-ai)) - by Maxim Salnikov
- [RapidChat](https://sushanthr.github.io/RapidChat/) ([source](https://github.com/sushanthr/RapidChat)) - by Sushanth Rajasankar
- [SD Turbo Image-to-Image](https://eyaler.github.io/webnn-developer-preview/demos/sd-turbo/) ([source](https://github.com/eyaler/webnn-developer-preview/)) by Eyal Gruss
- [Super Resolution](https://sushanthr.github.io/RapidEsrGan/) ([source](https://github.com/sushanthr/RapidEsrGan)) - by Sushanth Rajasankar
- [WebNN API Demo for Golang](https://me.sansmoraxz.com/webnngo-demo/) ([source](https://github.com/sansmoraxz/webnngo-demo)) - by Souyama
- [WebNN via Transformers.js / Angular](https://ng-ai-zeta.vercel.app/) ([source](https://github.com/webmaxru/ng-ai/)) - image classification, sentiment analysis, feature detection, offline-ready, installable (PWA), web workers - by [Maxim Salnikov](https://www.linkedin.com/in/webmax/)
- [WebNN via Transformers.js / Next.js](https://nextjs-webnn.vercel.app/) ([source](https://github.com/webmaxru/nextjs-webnn/)) - image classification - by Maxim Salnikov

## Frameworks

- [ONNX Runtime Web](https://onnxruntime.ai/)
  - [WebNN Supported Versions](https://onnxruntime.ai/docs/get-started/with-javascript/web.html#supported-versions)
- [Transformers.js](https://huggingface.co/docs/transformers.js) by Joshua Lochner
- [Web AI Toolkit](https://github.com/jgw96/web-ai-toolkit) by Justin Willis

## Presentations

- 2024.11 [Privacy-first in-browser Generative AI web apps: offline-ready, future-proof, standards-based](https://www.slideshare.net/slideshow/privacy-first-in-browser-generative-ai-web-apps-offline-ready-future-proof-standards-based/273142915) - by [Maxim Salnikov](https://www.linkedin.com/in/webmax/)
- 2024.07 [Generative AI power on the web: making web apps smarter with WebGPU and WebNN](https://www.thinktecture.com/contributions/generative-ai-power-on-the-web-making-web-apps-smarter-with-webgpu-and-webnn/) - by Christian Liebel
- 2024.02 [WebNN: Die AI-Revolution im Browser?](https://basta.net/web-development/webbnn-api-ai-browser/) - by Christian Liebel
- 2023.11 [第六届 FEDAY: WEBNN, WEB 端侧推理的未来](https://ibelem.github.io/webnn-is-the-future/) - by Belem Zhang
- 2023.10 [WebNN Implementation on DirectML](https://docs.google.com/presentation/d/1u9efG33BCIp0VdvpXXAu1yJlW10YvNNbnSklwJPBeCM/edit#slide=id.g24dab4effb5_0_0) - BlinkOn 18 - by Chai Chaoweeraprasit, Rafael Cintron, Ningxin Hu
- 2023.06 [W3C 中国 Web 前沿技术论坛: WebNN Updates](https://ibelem.github.io/webnn-updates/) [PDF / 简体中文](https://www.w3.org/2023/06/china-web-forum/slides/zhang-min.pdf) - by Belem Zhang

## Samples

### Samples on CPU or GPU

- [WebNN Samples](https://webmachinelearning.github.io/webnn-samples-intro/) ([source](https://github.com/webmachinelearning/webnn-samples))
- [NNotepad - WebNN Playground](https://webmachinelearning.github.io/webnn-samples/nnotepad/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/nnotepad)) - by Joshua Bell
- [Code Editor](https://webmachinelearning.github.io/webnn-samples/code/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/code))
- [Face Recognition](https://webmachinelearning.github.io/webnn-samples/face_recognition/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/face_recognition)) - FaceNet, SSD MobileNet V2 Face
- [Facial Landmark Detection](https://webmachinelearning.github.io/webnn-samples/facial_landmark_detection/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/facial_landmark_detection)) - Face Landmark (SimpleCNN), SSD MobileNet V2 Face
- [Handwritten Digits Classification](https://webmachinelearning.github.io/webnn-samples/lenet/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/lenet)) - LeNet
- [Image Classification](https://webmachinelearning.github.io/webnn-samples/image_classification/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/image_classification)) - MobileNet v2, ResNet50 v2, SqueezeNet
- [Noise Suppression](https://webmachinelearning.github.io/webnn-samples/nsnet2/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/nsnet2)) - NSNet2
- [Noise Suppression](https://webmachinelearning.github.io/webnn-samples/rnnoise/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/rnnoise)) - RNNoise
- [Object Detection](https://webmachinelearning.github.io/webnn-samples/object_detection/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/object_detection)) - Tiny Yolo v2, SSD MobileNet v1
- [Selfie Segmentation](https://webmachinelearning.github.io/webnn-samples/selfie_segmentation/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/selfie_segmentation)) - Selfie Segmentation
- [Semantic Segmentation](https://webmachinelearning.github.io/webnn-samples/semantic_segmentation/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/semantic_segmentation)) - DeepLab v3
- [Style Transfer](https://webmachinelearning.github.io/webnn-samples/style_transfer/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/style_transfer)) - Fast Style Transfer

### Samples on NPU

- [Image Classification](https://webmachinelearning.github.io/webnn-samples/image_classification/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/image_classification)) - EfficientNet Lite 4, MobileNet v2, ResNet 50 v1 (coming soon)
- [Object Detection](https://webmachinelearning.github.io/webnn-samples/object_detection/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/object_detection)) - SSD MobileNet v1
- [Selfie Segmentation](https://webmachinelearning.github.io/webnn-samples/selfie_segmentation/) ([source](https://github.com/webmachinelearning/webnn-samples/tree/master/selfie_segmentation)) - Selfie Segmentation

## Spec

- [W3C Web Neural Network API](https://www.w3.org/TR/webnn/) - by Ningxin Hu, Chai Chaoweeraprasit, Dwayne Robinson.

## Testimonials

## Tools

- [Onnx2Text](https://github.com/fdwr/Onnx2Text) - Converts an ONNX ML model protobuf from/to text - by Dwayne Robinson
- [WebNN Netron](https://ibelem.github.io/netron/) - Show WebNN support status in Chromium for models opened in Netron - by Belem Zhang

### WebNN Model-to-Code Conversion

- [ONNX2WebNN](https://github.com/huningxin/onnx2webnn) - by Ningxin Hu
- [WebNN Code Generator](https://github.com/ibelem/webnn-code-generator/) - by Belem Zhang
- [WebNN Utilities / OnnxConverter](https://github.com/MicrosoftEdge/WebNNUtils/) - by Microsoft Edge team

Read [more details](https://webnn.io/en/learn/tutorials/webnn/vanillajs) for generating WebNN Vanilla JavaScript code.

### WebNN Code-to-Code Translation

 - [To do] Converting existing Python-based ML code (particularly PyTorch/TorchScript) from other frameworks to WebNN lean vanilla JavaScript.

## Tutorials

### ONNX Runtime Web

- 2024.05 [Microsoft Learn: Windows AI / DirectML: WebNN Overview](https://learn.microsoft.com/en-us/windows/ai/directml/webnn-overview) ([简体中文](https://learn.microsoft.com/zh-cn/windows/ai/directml/webnn-overview)) ([繁體中文](https://learn.microsoft.com/zh-tw/windows/ai/directml/webnn-overview)) ([日本語](https://learn.microsoft.com/ja-jp/windows/ai/directml/webnn-overview))
- 2024.05 [Microsoft Learn: Windows AI / DirectML: WebNN API Tutorial](https://learn.microsoft.com/en-us/windows/ai/directml/webnn-tutorial) ([简体中文](https://learn.microsoft.com/zh-cn/windows/ai/directml/webnn-tutorial)) ([繁體中文](https://learn.microsoft.com/zh-tw/windows/ai/directml/webnn-tutorial)) ([日本語](https://learn.microsoft.com/ja-jp/windows/ai/directml/webnn-tutorial))
- [ONNX Runtime Web Tutorials](https://onnxruntime.ai/docs/tutorials/web/)
- [Build ONNX Runtime Web with WebNN Support](https://onnxruntime.ai/docs/build/web.html)
- [WebNN Operators Support Table](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webnn-operators.md) - by Wanming Lin

### WebNN API

- [Introduction to Web Neural Network API (WebNN)](https://webmachinelearning.github.io/get-started/2024/05/16/introduction-to-web-neural-network-api.html) - by Paul Cooper
- [Build Your First Graph with WebNN API](https://webmachinelearning.github.io/get-started/2021/03/15/build-your-first-graph-with-webnn-api.html)
- [Noise Suppression Net 2 (NSNet2)](https://webmachinelearning.github.io/get-started/2021/03/17/noise-suppression-net-v2.html)

## Videos

- 2025.06 [Smarter Angular Apps with WebNN & Prompt API](https://www.youtube.com/watch?v=s1xizqmdX4g) by Christian Liebel
- 2025.04 [BlinkOn 20: Compute Abstraction for AI: Wasm, WebGPU, and WebNN](https://www.youtube.com/watch?v=IgIdayJH4_o) by Phillis Tang
- 2025.03 [AI in Browser using WebNN](https://www.youtube.com/watch?v=qLDq3kj79DY) by Moh Haghighat and Guy Tamir
- 2024.11 [Web AI on next generation AI PCs](https://www.youtube.com/watch?v=5BjB7AIed3A) by Moh Haghighat
- 2024.11 [The Web Neural Network (WebNN) API: Where we are and what's Next](https://www.youtube.com/watch?v=FoYBWzXCsmM) by Rob Kochman, Rafael Cintron
- 2024.11 [Empowering AI PC’s with Intel’s AI Software Stack](https://www.youtube.com/watch?v=j_g1IKIDqHs) by Prashant Bhardwaj
- 2024.10 [Web Neural Network (WebNN) API Workshop](https://www.youtube.com/watch?v=euQj7VHQk3Y) [อักษรไทย] by [Surasuk Oakkharaamonphong](https://twitter.com/surasuk612)
- 2024.07 [Innovations in WebNN](https://www.youtube.com/watch?v=JwLtMFS_2UE) by Jerry Makare
- 2024.05 [Microsoft Build '24: Bring AI experiences to all your Windows Devices](https://build.microsoft.com/en-US/sessions/65c11f47-56d8-442b-ae52-48df62b7b542) by Adele Parsons
- 2024.05 [Microsoft Build '24: The Web is AI Ready—maximize your AI web development with WebNN](https://build.microsoft.com/en-US/sessions/fe8f0c03-6f31-400a-8954-4e37c935e6e9) - by Moh Haghighat
- 2024.05 [Web Neural Networks for the AI PC](https://www.youtube.com/watch?v=kpJRfm5tunQ) ([bilibili](https://www.bilibili.com/video/BV1Y1421B7SQ/)) - by Guy Tamir
- 2024.02 [QCon上海 2023: WebNN，Web 端侧推理的未来](https://www.infoq.cn/video/LXliFqOfOrj2wNvslzZ0) - by Ningxin Hu
- 2023.10 [Google BlinkOn 18: WebNN Implementation on DirectML](https://www.youtube.com/watch?v=FapumEVdrcg) - by Chai Chaoweeraprasit
- 2023.10 [AI @ W3C](https://www.youtube.com/watch?v=E0TbotgqAgw) by Dominique Hazael-Massieux
- 2021.12 [OpenCV Webinar 13: Chinese, Use WebNN to Optimize OpenCV.js DNN](https://www.youtube.com/watch?v=kQogwlhSsQ4) - by Hanxi Guo
- 2021.11 [Introducing WebNN as a new backend for TensorflowJS](https://www.youtube.com/watch?v=v3LAY-Do25I) - by Shivay Lamba
- 2021.10 [W3C TPAC 2021: WebNN Performance Comparison](https://www.youtube.com/watch?v=cHmWE5IHo9o&list=PLNhYw8KaLq2VOeJCyWZiEcmVYpXl5dw81) - by Wanming Lin

## Websites

- [W3C Web Neural Network](https://webnn.dev/)
- [WebNN Documentation](https://webnn.io/)
- [WebNN Developer Preview](https://microsoft.github.io/webnn-developer-preview)
- [WebNN: Intel AI PC Development](https://www.intel.com/content/www/us/en/developer/topic-technology/ai-pc/webnn.html)

## Bug Reporting

- [Chromium](https://issues.chromium.org/issues?q=status:open%20componentid:1456206&s=created_time:desc)

## Source Code

- [Chromium](https://source.chromium.org/chromium/chromium/src/+/main:services/webnn/)

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0)
