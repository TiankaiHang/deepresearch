NVIDIA TensorRT 深度学习推理加速架构原理


第一章：TensorRT 在 AI 推理生命周期中的角色

本章旨在为 NVIDIA TensorRT 建立基础背景，将其定位为解决“推理问题”的专业工具，并阐明其设计哲学和核心目标，区别于通用的深度学习训练框架。

1.1 定位 TensorRT：训练后优化器与运行时

NVIDIA TensorRT 是一个软件开发工具包 (SDK)，其核心包含一个高性能的深度学习推理优化器和一个运行时 1。它的首要职责是接收一个
已经训练完成的神经网络，并为其生成一个高度优化的“运行时引擎”(runtime engine)，专门用于生产环境的部署 3。
“训练后”(post-training) 这一特性是其关键的设计决策。与 PyTorch 或 TensorFlow 等既要支持训练（需要灵活的动态图结构以进行反向传播）又要兼顾基本推理的框架不同，TensorRT 完全专注于推理过程 6。这种专一性使其能够对计算图执行激进的、甚至是破坏性的优化，而这些优化在训练阶段是不可行的。
TensorRT 的编译过程最终产物是一个序列化的计划文件（plan file），即引擎。这是一个自包含的、可部署的实体，能够在 NVIDIA 硬件上以极低的延迟和极高的吞吐量执行推理 3。
这种设计理念体现了一个根本性的权衡：牺牲训练所需的灵活性，以换取在特定、固定的硬件目标上实现极致的推理性能。它将现代机器学习运维 (MLOps) 流程明确地划分为“开发/训练”和“生产/推理”两个阶段，而 TensorRT 正是连接这两个阶段的关键桥梁。从本质上说，TensorRT 将神经网络从一个用于研究和实验的数学模型，转变为一个为特定任务和硬件编译和优化的、高效的软件程序。

1.2 推理问题：延迟、吞吐量与效率

部署现代深度学习模型面临着严峻的挑战。庞大而复杂的模型架构在计算上是昂贵的，这使得在对延迟敏感的应用场景（如自动驾驶、实时视频分析）中实现实时推理成为一个巨大的障碍 6。
本报告将深入探讨两个关键性能指标：延迟 (Latency)，即处理单个输入所需的时间；以及吞吐量 (Throughput)，即单位时间内完成的推理次数 9。TensorRT 的工程目标就是最小化前者并最大化后者 3。其应用场景广泛，从超大规模数据中心到资源受限的边缘设备和车载平台，每个场景都有其独特的功耗和计算预算，而 TensorRT 正是为了应对这些多样化的需求而设计的 3。

1.3 TensorRT 生态系统：推理优化的整体解决方案

TensorRT 并非一个孤立的库，而是一个“工具生态系统” 9。这个生态系统体现了 NVIDIA 为整个推理流程提供端到端完整解决方案的战略。它认识到，单一的编译器无法对所有模型架构和部署场景都做到最优。
该生态系统的关键组成部分包括：
核心 C++/Python API：用于构建网络和运行推理引擎的基础库 4。
解析器 (Parsers)：主要是 ONNX 解析器，作为从 PyTorch、TensorFlow 等主流框架导入模型的主要入口 13。
TensorRT Model Optimizer：一个统一的库，集成了量化、剪枝、稀疏化等模型压缩技术，为模型在下游框架中的高效部署做准备 10。
TensorRT-LLM：一个专门的开源库，通过定制化的内核和超越通用编译器能力的优化来加速大型语言模型 (LLM) 10。
与部署解决方案的集成：与 NVIDIA Triton 推理服务器等服务软件无缝集成，后者增加了动态批处理、并发模型执行和模型集成等生产级功能 3。
这个生态系统的构建是 NVIDIA 的一项战略举措，旨在为其硬件提供一个全面、垂直整合的 AI 推理解决方案。一方面，它通过为特定问题提供专业化工具来提高开发者的生产力；另一方面，它也加深了开发者对 NVIDIA 软件栈的依赖。实现顶尖性能，尤其是对于大型语言模型，不再是使用单个工具那么简单，而更多地是采纳 NVIDIA 预设的整个工作流程。

第二章：核心优化原理：从计算图到优化引擎

本章是报告的技术核心，将深入剖析 TensorRT 在引擎构建阶段所采用的主要优化技术。这个过程可以类比于传统软件的编译，将高级的、人类可读的描述（神经网络图）转化为针对特定硬件的高度优化的、低级的可执行文件（TensorRT 引擎）。

2.1 图优化：剪枝与重构

TensorRT 的优化之旅始于将输入模型（例如，一个 ONNX 文件）解析为其内部的图表示 3。随后，它会执行一系列与具体硬件目标无关的图级别转换，包括垂直和水平层融合以及层消除。
无用操作消除：图中对最终输出没有贡献的操作将被彻底剪除。
操作聚合：具有相同参数的连续操作可以被聚合。
子图划分：在与框架集成（如 TF-TRT 16 或 MXNet-TensorRT 18）的场景中，TensorRT 会识别出图中由其支持的操作组成的连续子图。这些子图随后被替换为单个的
TRTEngineOp 节点，形成一个混合图。在这种模式下，原始框架负责执行不支持的操作，并在遇到 TRTEngineOp 节点时调用 TensorRT 引擎来执行优化后的部分。

2.2 层与张量融合：最小化内核开销与内存访问

原理：这是 TensorRT 最关键的优化之一。它将多个独立的层（例如，一个卷积层、一个偏置加法层和一个 ReLU 激活层）合并成一个单一的、手工调优的 CUDA 内核 3。
益处 1：降低内存带宽压力：如 21 中所述，若不进行融合，每一层的输出（即中间张量）都必须写入到速度较慢的全局 GPU 内存 (DRAM) 中，然后再被下一层读取。这是一个耗时的过程。融合技术使得中间结果能够保留在速度快得多的片上内存（寄存器或 L1/L2 缓存）中，并被同一内核内的后续操作立即消耗，从而极大地减少了内存流量 7。这一优化是针对“内存墙”问题的直接攻击——现代 GPU 的计算能力（FLOPS）增长速度远超内存带宽的增长速度，导致计算单元常常因等待数据而闲置。层融合通过将中间数据保留在内存层次结构的最顶端，确保了计算核心能够持续获得数据，从而最大化其利用率。
益处 2：减少内核启动开销：每次启动 CUDA 内核都会产生微小的 CPU 开销。通过将 N 个层融合成一个，TensorRT 减少了 N-1 次内核启动，这在拥有数千个层的深度模型中会累积成显著的性能提升 21。
插件机制应对高级融合：TensorRT 的自动化模式匹配编译器能够识别并融合许多常见的操作序列。然而，对于像 Flash-Attention 这样极其复杂的、前沿的模式，自动发现是不可行的 21。针对这些情况，TensorRT 提供了一个 C++
插件 (Plugin) 接口。插件是用户定义的、自定义编写的内核，它们被注册到 TensorRT 中并插入到计算图中。这使得 NVIDIA（在其 TensorRT-LLM 等库中）或最终用户能够实现超越编译器当前能力的、新颖且高度复杂的融合方案 2。

2.3 内核自动调优：基于实证的性能优化

原理：对于任何给定的操作（例如，一个特定的卷积），都存在多种算法实现（即 CUDA 内核）。最优选择在很大程度上取决于该层的参数（如滤波器大小、步长等）、输入数据的维度、所选的精度以及具体的目标 GPU 架构（如图灵、安培或霍珀架构）7。
过程：TensorRT 并不使用单一的通用内核，而是维护着一个庞大的、包含多种高度优化内核实现的库。在引擎构建过程中，它会进行一次实证搜索：直接在目标 GPU 上对每个层的多种内核“策略 (tactics)”进行基准测试，以找出执行速度最快的那一个 3。
这个过程将 TensorRT 引擎的构建类比于为特定 GPU “预编译和静态链接”一个二进制可执行文件。一个通用的程序被编译成针对特定 CPU 架构（如 x86-64）的机器码。TensorRT 的构建过程与此类似，它将一个高级程序描述（神经网络图）“编译”成一个高度优化的二进制文件（引擎）。这里的“机器”不仅仅是一个 GPU 架构，而是特定的 GPU 型号、批处理大小和精度。自动调优器的实证搜索就像传统编译器中的性能分析引导优化 (PGO) 过程，但更为详尽。最终的引擎不仅包含了图结构，还包含了在搜索过程中选定的、最快的 CUDA 内核的指针及其预先确定的最优启动参数。
工作空间与时间缓存：这个基准测试过程需要临时的 GPU 内存，被称为工作空间 (workspace) 15。工作空间的大小会影响到可以被考虑的策略范围。由于这个过程可能非常耗时，TensorRT 可以创建一个
时间缓存 (timing cache) 来存储这些基准测试的结果，从而极大地加快后续在相同硬件上构建相同或相似模型的速度 27。

2.4 动态张量内存管理

原理：TensorRT 对整个计算图进行存活分析 (liveness analysis)，以理解每个中间张量（激活值）的“生命周期”。
内存复用：随后，它会分配一个内存池，并为生命周期不重叠的张量智能地复用内存缓冲区 3。它不会为每个张量都分配新的缓冲区，而是重新利用那些在后续计算中不再需要的“死亡”张量的内存。
益处：这项技术显著降低了模型在推理过程中的峰值 GPU 内存占用，这部分占用通常由激活值主导 20。这使得更大的模型或更大的批处理大小能够适应给定的显存容量 28。

第三章：精度校准与量化数学

本章将深入探讨 TensorRT 如何利用低精度计算，解释其背后的数学原理以及为保持准确性所需的过程。

3.1 低精度计算的理论依据

推理过程通常不需要训练时所使用的高数值精度（FP32）1。使用如 FP16、BF16、INT8、FP8 或 INT4 等低精度格式主要带来两大好处：
减少内存占用：存储权重和激活值需要更少的内存和带宽 3。
加速计算：NVIDIA GPU 包含专门的硬件单元（张量核心，Tensor Cores），它们执行低精度类型矩阵运算的速度远快于 FP32 25。

3.2 从 FP32 到低精度格式

FP16/BF16：半精度浮点格式，在数值范围和精度之间取得了良好平衡，通常能在几乎不损失精度的情况下带来“免费”的性能提升 6。
INT8：8 位整数量化能提供最显著的加速，但需要一个精细的校准 (calibration) 过程来维持准确性 3。
FP8/FP4 (E4M3, E5M2)：在 Hopper、Blackwell 及更新的 GPU 上支持的新格式 29。它们比 INT8 提供更宽的动态范围，因此更适合处理像 Transformer 这样具有较大数值分布的模型，但代价是精度较低 30。硬件和软件的演进与模型架构的发展是同步的。随着新模型类型的普及，新的数据格式和硬件支持也应运而生。向 FP8 的转变正是 Transformer 架构占据主导地位的直接结果。

3.3 训练后量化 (PTQ) 与校准过程

对称量化原理：TensorRT 采用对称量化方案，其中浮点值 x 通过一个单一的缩放因子 s 映射到量化整数 xq​：xq​=round(clip(x/s)) 31。核心挑战在于为每个张量找到最优的缩放因子
s。
校准器的角色 (IInt8Calibrator)：对于训练后量化 (PTQ)，用户必须提供 IInt8Calibrator 接口的实现 31。
校准数据集：向校准器提供一小批但具有代表性的输入数据 15。
激活值直方图：TensorRT 在此数据集上运行 FP32 模型，并为网络中的每个张量收集激活值的直方图。
最小化信息损失：校准器的算法（例如 IInt8EntropyCalibrator2）分析这些直方图以确定一个裁剪阈值。选择此阈值的目的是在截断误差（裁剪掉那些会过度拉伸量化范围的大幅值离群点）和离散化误差（损失了范围内数值的分辨率）之间找到最佳平衡。这个过程旨在最小化 FP32 分布与量化后分布之间的 Kullback-Leibler (KL) 散度，从而保留信息并维持准确性 31。
校准是一个统计优化问题，而非简单的线性缩放。一个幼稚的量化方法是找到校准数据中的 min 和 max 值，然后将 INT8 的范围 [-127, 127] 映射到 [-max(abs(min), abs(max)), +max(abs(min), abs(max))]。然而，神经网络的激活值通常呈现长尾分布，带有少数极端离群值。使用绝对最大值会导致绝大多数数值被压缩到整数范围的一个极小部分，引发巨大的离散化误差和精度下降。TensorRT 基于熵的校准是一种更复杂的统计方法，它通过刻意裁剪离群值（接受一定的截断误差）来为分布的主体部分提供更高的分辨率和更低的离散化误差。
校准缓存：这个过程的产物是一个“校准表”，其中包含了为每个张量计算出的缩放因子。该表可以被缓存到磁盘，这样耗时的校准过程在后续引擎构建时就不需要重复进行 31。

3.4 量化感知训练 (QAT)

作为 PTQ 的替代方案，量化感知训练 (QAT) 在训练过程中模拟量化的影响 8。这是通过在训练图中插入“伪”量化和反量化节点来实现的。这使得模型的权重能够适应精度损失，通常能为量化后的模型带来比 PTQ 更高的最终准确性，特别是对于那些对量化非常敏感的模型。

第四章：TensorRT 工作流程与部署策略

本章将提供一个使用 TensorRT 的实践指南，从模型转换到部署最终引擎，并重点关注如何处理可变输入尺寸等真实世界的复杂情况。

4.1 标准工作流程：从框架到引擎

第一步：导出为 ONNX：最常见且推荐的第一步是从原始框架（如 PyTorch、TensorFlow）将训练好的模型导出为开放神经网络交换 (ONNX) 格式 1。ONNX 充当了一个通用的中间表示。然而，这个过程并非总是万无一失。训练框架发展迅速，不断引入新的算子和行为，而 ONNX 标准和特定框架的导出器必须不断追赶。这常常导致不匹配，例如模型使用了尚不支持的算子，或是在导出时被错误转换。因此，开发者通常需要使用
Polygraphy 和 ONNX-GraphSurgeon 等工具来调试和修复导出的 ONNX 图 13。
第二步：构建 TensorRT 引擎：将 ONNX 文件传递给 TensorRT 构建器 (builder)。这可以通过以下方式完成：
使用 trtexec 命令行工具进行快速基准测试和引擎生成 33。
使用 TensorRT Python 或 C++ API 对构建过程进行编程控制 1。在此步骤中，需要配置关键参数，如精度（FP16、INT8）、工作空间大小以及用于动态形状的优化配置文件。
第三步：序列化和部署引擎：构建器生成一个序列化的引擎文件。该文件最终由应用程序中的 TensorRT 运行时加载以执行推理 36。

4.2 处理输入可变性：动态形状

挑战：许多应用要求模型能够处理不同维度的输入，例如变化的批处理大小或图像分辨率。静态尺寸的引擎不适用于这些情况。
解决方案：优化配置文件：TensorRT 通过动态形状 (Dynamic Shapes) 功能来解决这个问题 37。在构建引擎时，用户可以定义一个或多个
IOptimizationProfile。对于每个动态输入，配置文件需要指定：
min：可能的最小维度。
opt：典型或最优的维度。
max：可能的最大维度。
工作原理：TensorRT 使用 opt 维度进行其内核自动调优过程，以生成针对最常见用例的最高性能内核。同时，它也会创建一个能够正确执行 min 和 max 范围内任何输入尺寸的计划，但对于非最优形状，可能会使用性能较低的通用内核 7。这体现了性能与灵活性之间的直接权衡。为了最大化性能，最佳实践是在应用允许的情况下始终使用静态形状。
两阶段执行：带有动态形状的引擎采用一种“乒乓”执行模型。形状计算在 CPU 上进行。如果某个形状的计算依赖于 GPU 计算的结果，就会发生同步，数据可能需要被复制回 CPU，从而产生开销 37。

4.3 部署范式

独立运行时：应用程序直接使用 TensorRT C++ 或 Python 运行时 API 来加载引擎并执行推理。这种方式开销最低，控制力最强，非常适合性能关键型应用，如车载系统或机器人 1。
框架集成部署 (TF-TRT, Torch-TensorRT)：将 TensorRT 用作更大 TensorFlow 或 PyTorch 图中的加速器。这种方法非常方便，允许开发者留在熟悉的框架内，同时在兼容的子图上获得显著的速度提升。框架负责处理执行流程，为优化后的节点调用 TensorRT 10。
规模化服务 (NVIDIA Triton 推理服务器)：对于数据中心和云部署，TensorRT 引擎通常通过 Triton 提供服务。Triton 是一个专用的推理服务软件，提供了一个生产就绪的环境，具备 HTTP/gRPC 端点、动态批处理（将单个请求组合成更大的批次以提高 GPU 利用率）、并发模型执行和健康监控等功能 3。

第五章：对比分析与战略建议

本章将 TensorRT 置于更广泛的 AI 推理技术版图中，通过与主要竞争对手的清晰比较，为从业者提供战略性建议。

5.1 TensorRT 与竞争性运行时的技术比较

本节将对 TensorRT、英特尔的 OpenVINO 和跨平台的 ONNX Runtime 进行详细的正面比较。分析将围绕以下关键维度展开：
硬件特异性：TensorRT 专为 NVIDIA GPU 设计 38。OpenVINO 专为英特尔硬件（CPU、集成 GPU、VPU）优化 38。ONNX Runtime 是硬件无关的，通过执行提供程序 (Execution Providers, EPs) 来支持不同的后端，包括 TensorRT 和 OpenVINO 3。
优化哲学：TensorRT 的方法与硬件深度集成，依赖于对 GPU 架构的专有知识来执行激进的、基于实证的优化，如针对张量核心的内核自动调优 7。OpenVINO 的方法更侧重于其目标硬件的软件层面，专注于 CPU 级别的优化（如 AVX 指令集向量化），并利用其模型优化器创建可移植的中间表示 (IR) 38。
性能与用例：在 NVIDIA GPU 上，由于其硬件特异性设计，TensorRT 通常能提供最高的性能和最低的延迟 3。OpenVINO 在英特尔 CPU 上表现优越，专为英特尔硬件上的边缘和物联网应用量身定制 38。ONNX Runtime 提供了最大的可移植性，但其在特定平台上的性能完全取决于其后端 EP 的质量；它通常充当一个方便的包装器，而不是性能顶尖的独立解决方案 39。
“最佳”推理引擎的选择完全取决于具体情境。不存在一个普遍“最好”的推理引擎；选择从根本上受到目标部署硬件的制约。如果目标是 NVIDIA DGX 服务器，TensorRT 是合乎逻辑且性能最高的选择。如果目标是边缘端的英特尔 NUC，OpenVINO 则是正确的选择。如果一个组织需要将同一模型部署到使用 NVIDIA GPU 的云环境以及本地的英特尔 CPU 服务器，那么 ONNX Runtime 提供了一个统一的 API，它抽象了硬件特定的后端。
特性
NVIDIA TensorRT
Intel OpenVINO
ONNX Runtime
主要硬件目标
NVIDIA GPU (数据中心, 工作站, 边缘) 38
Intel CPU, 集成 GPU, VPU 38
跨平台 (CPU, GPU 等) 通过执行提供程序 (EPs) 3
优化重点
深度硬件集成, 内核自动调优, 激进的层融合, 利用张量核心 7
以 CPU 为中心的优化 (AVX 指令), 基于软件的技术, 异构执行 38
通用目的; 依赖于像 TensorRT 或 OpenVINO 这样的 EP 来实现硬件特定加速 3
输入模型格式
ONNX, TensorFlow, PyTorch (通过解析器/集成) 13
ONNX, TF, PyTorch, Caffe (通过模型优化器转为中间表示 - IR) 38
ONNX (原生格式) 14
输出产物
序列化引擎 (.plan), 特定于 GPU 架构和 TRT 版本 27
中间表示 (.xml, .bin), 可在英特尔硬件间移植 38
ONNX 模型本身用于执行
核心优势
在 NVIDIA GPU 上实现最高性能 3
在广泛的英特尔硬件上提供优化性能, 特别是边缘/CPU 38
最大的可移植性和框架互操作性 14
主要应用场景
延迟关键型应用, 高吞吐量数据中心, NVIDIA 硬件上的自动驾驶系统 3
基于英特尔的边缘设备和服务器上的计算机视觉和深度学习 38
使用单一模型格式和 API 在多样化硬件上部署模型


5.2 最大化性能的最佳实践

模型设计：优先选择那些已知能被 TensorRT 良好支持和融合的操作与模式。
精度选择：始终从 FP16 开始，因为它通常能以很小的代价提供显著的速度提升 18。为追求极致性能，投入精力进行 INT8 校准，并确保使用高质量、有代表性的校准数据集。对于现代 Transformer，应评估 FP8 的效果。
形状处理：尽可能使用静态形状。如果必须使用动态形状，应仔细定义优化配置文件，使其反映最常见的运行时输入 6。
性能分析：使用 trtexec 和 NVIDIA Nsight Systems 等工具对引擎进行性能分析，以识别逐层的性能瓶颈 6。这可以揭示某些层是否未被融合或回退到了较慢的实现。
批处理：在 GPU 显存允许的范围内最大化批处理大小以提高吞吐量，尤其是在服务器端部署中。利用 Triton 的动态批处理功能为实时请求流自动化此过程。

5.3 未来展望：推理优化的演进

未来的趋势包括由生成式 AI 和 LLM 的巨大规模驱动的低比特格式（FP8, FP4, INT4）日益增长的重要性 29。同时，AI 在优化自身方面的作用也日益凸显，例如使用 LLM 自动生成优化的 CUDA 内核，这是 NVIDIA 正在积极探索的研究领域 45。新模型架构与编译器优化之间的持续博弈也将继续，新的模式将需要新的融合策略，并可能催生新的插件。

结论

NVIDIA TensorRT 是一套高度专业化的工具，其核心加速原理根植于对深度学习推理过程的深刻理解和对 NVIDIA GPU 硬件架构的极致利用。它通过一系列从图重构到硬件特定内核选择的、激进的编译时优化，将一个通用的训练后模型转化为一个高效的、专用的推理引擎。
其关键加速机制包括：
图级优化与层融合：通过合并多个操作层为一个单一的 CUDA 内核，从根本上减少了对慢速全局内存的访问次数和内核启动开销，这是其最重要的性能来源。
低精度量化：利用 FP16、INT8 和 FP8 等低精度格式，借助 GPU 上的专用张量核心硬件实现计算加速，并通过精密的校准过程来维持模型准确性。
内核自动调优：在目标硬件上进行实证基准测试，为每个操作选择最快的内核实现，从而生成一个高度定制化的、非可移植的引擎。
高效内存管理：通过智能地复用张量内存，最小化推理过程中的峰值内存占用。
TensorRT 的设计哲学决定了它在性能上的领先地位，同时也限定了其应用范围。它在 NVIDIA 生态系统内提供了无与伦比的性能，但这种性能是以牺牲跨平台可移植性为代价的。对于追求在 NVIDIA 硬件上实现最低延迟和最高吞吐量的开发者而言，深入理解并掌握 TensorRT 的工作原理和工作流程，是释放其 AI 应用全部潜力的关键。随着模型日益复杂，TensorRT 及其生态系统将继续作为连接 AI 模型研发与高性能生产部署之间不可或缺的桥梁。
引用的著作
Quick Start Guide — NVIDIA TensorRT Documentation, 访问时间为 六月 23, 2025， https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html
TensorRT Deployment — mmcv 1.4.1 documentation - Read the Docs, 访问时间为 六月 23, 2025， https://mmcv.readthedocs.io/en/v1.4.1/deployment/tensorrt_plugin.html
TensorRT: NVIDIA Deep Learning Optimizer - Ultralytics, 访问时间为 六月 23, 2025， https://www.ultralytics.com/glossary/tensorrt
Overview — NVIDIA TensorRT Documentation - NVIDIA Docs Hub, 访问时间为 六月 23, 2025， https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/overview.html
5. ROS1-ROS+Deep Learning Basic Lesson — JetArm v1.0 documentation - Hiwonder Docs, 访问时间为 六月 23, 2025， https://docs.hiwonder.com/projects/JetArm/en/latest/docs/5.ROS1_Deep_Learning_Basic_Lesson.html
Accelerating AI Inference With TensorRT - DZone, 访问时间为 六月 23, 2025， https://dzone.com/articles/accelerate-ai-interference-tensorrt
From PyTorch to Petawatts - ABINASH KUMAR MISHRA, 访问时间为 六月 23, 2025， https://hustlercoder.substack.com/p/from-pytorch-to-petawatts
TensorRT Implementations of Model Quantization on Edge SoC - Computer Science : Texas State University, 访问时间为 六月 23, 2025， https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/mcsoc23.pdf
developer.nvidia.com, 访问时间为 六月 23, 2025， https://developer.nvidia.com/tensorrt#:~:text=NVIDIA%C2%AE%20TensorRT%E2%84%A2%20is,high%20throughput%20for%20production%20applications.
TensorRT SDK | NVIDIA Developer, 访问时间为 六月 23, 2025， https://developer.nvidia.com/tensorrt
Model inference using TensorFlow and TensorRT - Databricks Documentation, 访问时间为 六月 23, 2025， https://docs.databricks.com/aws/en/machine-learning/model-inference/resnet-model-inference-tensorrt
NVIDIA TensorRT Documentation, 访问时间为 六月 23, 2025， https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
Overview — NVIDIA TensorRT Documentation, 访问时间为 六月 23, 2025， https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html
Runtime Optimization of ONNX Models With TensorRT, 访问时间为 六月 23, 2025， https://dev.co/runtime-optimization-of-onnx-models-with-tensorrt
TensorRT Export for YOLO11 Models - Ultralytics YOLO Docs, 访问时间为 六月 23, 2025， https://docs.ultralytics.com/integrations/tensorrt/
Announcing TensorRT integration with TensorFlow 1.7 - Google Developers Blog, 访问时间为 六月 23, 2025， https://developers.googleblog.com/en/announcing-tensorrt-integration-with-tensorflow-17/
Introducing TensorFlow with TensorRT (TF-TRT) - IBM Developer, 访问时间为 六月 23, 2025， https://developer.ibm.com/tutorials/introducing-tensorflow-with-tensorrt
Optimizing Deep Learning Computation Graphs with TensorRT - Apache MXNet, 访问时间为 六月 23, 2025， https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/tensorrt/tensorrt.html
Optimizing Deep Learning Computation Graphs with TensorRT - Apache MXNet, 访问时间为 六月 23, 2025， https://mxnet.apache.org/versions/1.5.0/tutorials/tensorrt/inference_with_trt.html
Optimizing Deep Learning Models with TensorRT - AST Consulting, 访问时间为 六月 23, 2025， https://astconsulting.in/artificial-intelligence/ml-machine-learning/optimizing-deep-learning-models-with-tensorrt
Model Definition — TensorRT-LLM - GitHub Pages, 访问时间为 六月 23, 2025， https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html
onnx-tensorrt/docs/faq.md at 10.9-GA - GitHub, 访问时间为 六月 23, 2025， https://github.com/onnx/onnx-tensorrt/blob/10.9-GA/docs/faq.md
TensorRT-LLM: A Comprehensive Guide to Optimizing Large Language Model Inference for Maximum Performance - Unite.AI, 访问时间为 六月 23, 2025， https://www.unite.ai/tensorrt-llm-a-comprehensive-guide-to-optimizing-large-language-model-inference-for-maximum-performance/
3 Inference Engines for optimal model throughput - Substack, 访问时间为 六月 23, 2025， https://substack.com/home/post/p-146886554?utm_campaign=post&utm_medium=web
What are the benefits of using TensorRT for inference on NVIDIA GPUs? - Massed Compute, 访问时间为 六月 23, 2025， https://massedcompute.com/faq-answers/?question=What%20are%20the%20benefits%20of%20using%20TensorRT%20for%20inference%20on%20NVIDIA%20GPUs?
ultralytics/docs/en/integrations/tensorrt.md at main - GitHub, 访问时间为 六月 23, 2025， https://github.com/ultralytics/ultralytics/blob/main/docs/en/integrations/tensorrt.md
End-to-End AI for NVIDIA-Based PCs: NVIDIA TensorRT Deployment, 访问时间为 六月 23, 2025， https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-nvidia-tensorrt-deployment/
How does TensorRT's dynamic memory allocation impact model performance?, 访问时间为 六月 23, 2025， https://massedcompute.com/faq-answers/?question=How%20does%20TensorRT%27s%20dynamic%20memory%20allocation%20impact%20model%20performance?
NVIDIA TensorRT for RTX Introduces an Optimized Inference AI Library on Windows 11, 访问时间为 六月 23, 2025， https://developer.nvidia.com/blog/nvidia-tensorrt-for-rtx-introduces-an-optimized-inference-ai-library-on-windows/
Optimizing Transformer-Based Diffusion Models for Video Generation with NVIDIA TensorRT, 访问时间为 六月 23, 2025， https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/
Working with Quantized Types — NVIDIA TensorRT Documentation, 访问时间为 六月 23, 2025， https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
TensorRT: Creating An ONNX Model - ccoderun.ca, 访问时间为 六月 23, 2025， https://www.ccoderun.ca/programming/doxygen/tensorrt/md_TensorRT_tools_onnx-graphsurgeon_examples_01_creating_a_model_README.html
How TensorRT Video Analytics Revolutionized Our Pipeline - KeyValue, 访问时间为 六月 23, 2025， https://www.keyvalue.systems/blog/from-bottlenecks-to-breakthroughs-how-tensorrt-video-analytics-revolutionized-our-pipeline/
Tutorial 9: ONNX to TensorRT (Experimental) - MMDetection's documentation!, 访问时间为 六月 23, 2025， https://mmdetection.readthedocs.io/en/v2.18.0/tutorials/onnx2tensorrt.html
Converting a PyTorch ONNX model to TensorRT engine - Jetson Orin Nano - Stack Overflow, 访问时间为 六月 23, 2025， https://stackoverflow.com/questions/78787534/converting-a-pytorch-onnx-model-to-tensorrt-engine-jetson-orin-nano
NVIDIA Developer How To Series: Introduction to Recurrent Neural Networks in TensorRT, 访问时间为 六月 23, 2025， https://www.youtube.com/watch?v=G3QA3ZzD4oc
Working with Dynamic Shapes — NVIDIA TensorRT Documentation, 访问时间为 六月 23, 2025， https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html
What are the key differences between TensorRT and OpenVINO in terms of model optimization? - Massed Compute, 访问时间为 六月 23, 2025， https://massedcompute.com/faq-answers/?question=What%20are%20the%20key%20differences%20between%20TensorRT%20and%20OpenVINO%20in%20terms%20of%20model%20optimization?
Clarity needed on differences between acceleration frameworks/runtimes for AGX Xavier, 访问时间为 六月 23, 2025， https://forums.developer.nvidia.com/t/clarity-needed-on-differences-between-acceleration-frameworks-runtimes-for-agx-xavier/168790
What are the key differences between TensorRT and OpenVINO in terms of performance?, 访问时间为 六月 23, 2025， https://massedcompute.com/faq-answers/?question=What+are+the+key+differences+between+TensorRT+and+OpenVINO+in+terms+of+performance%3F
What are the key differences between NVIDIA TensorRT and OpenVINO in terms of model optimization and inference performance? - Massed Compute, 访问时间为 六月 23, 2025， https://massedcompute.com/faq-answers/?question=What%20are%20the%20key%20differences%20between%20NVIDIA%20TensorRT%20and%20OpenVINO%20in%20terms%20of%20model%20optimization%20and%20inference%20performance?
Model Benchmarking with Ultralytics YOLO, 访问时间为 六月 23, 2025， https://docs.ultralytics.com/modes/benchmark/
Deep Learning Inference Frameworks Benchmark - arXiv, 访问时间为 六月 23, 2025， https://arxiv.org/pdf/2210.04323
Best Practices — NVIDIA TensorRT Documentation, 访问时间为 六月 23, 2025， https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling, 访问时间为 六月 23, 2025， https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/
