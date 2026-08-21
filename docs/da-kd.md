# DA-KD: Difficulty-Aware Knowledge Distillation for Efficient Large Language Models

1 2 1 2 1 3 1 2 **Changyi He** **Yifu Ding** **Jinyang Guo** **Ruihao Gong⁴ Haotong Qin⁵ Xianglong Liu**

34 Average Llama2-1.3b 32 Llama2-2.7b Qwen2.5-0.5b Qwen2.5-1.5b 30 28

26 Performance 24 22

0 100SeqKD Ours Iterations

| (min) SFT | KD-K L KD-RKL | GKD Distillm |
| --- | --- | --- |
| Time |  |  |
| Training |  |  |

Training Time 1963 200 300 Iterations 400 3570

*Figure 1.* Performance of various distillation methods on Dolly

dataset in the upper part, and their training time (minutes) and iterations in the bottom part.

## Abstract

Although knowledge distillation (KD) is an effec- tive approach to improve the performance of a smaller LLM (i.e., the student model) by trans- ferring knowledge from a large LLM (i.e., the teacher model), it still suffers from high training cost. Existing LLM distillation methods ignore the difficulty difference among different samples, making the distillation of easy samples unnec- essary. This leads to high distillation cost. In this paper, we propose difficulty-aware knowl- edge distillation (DA-KD) framework for efficient knowledge distillation, in which we dynamically adjust the distillation dataset based on the diffi- culty of samples. We further observe existing KD loss cannot perform well when most of samples are difficult in the distillation dataset because of unstable optimization and the neglect of hard sam- ples. Therefore, we also propose a new KD loss called bidirectional discrepancy loss (BDL) for effective KD. Extensive experiments demonstrate that our DA-KD framework is effective and ef- ficient. Without bells and whistles, DA-KD can outperform existing state-of-the-art KD methods by 2% with half training cost and even surpass the teacher model with 4.7*×* compression.

and storage requirements, posing challenges for practical deployment (Yang et al.,2024b). To address this issue, nu- merous model compression methods have emerged, such as quantization (Guo et al.,2024;Lv et al.,2024), pruning (Wang et al.,2024b;Guo et al.,2022;2023b) and knowl- edge distillation (Hinton,2015). Knowledge distillation is an effective approach for constructing smaller and more effi- cient neural networks. It aims to transfer knowledge from a high-performing teacher model to a compact student model for efficient inference.

Despite the success of KD for effective training of stu- dent models, it still suffers from high training cost, which normally requires hundreds of GPU hours when distill- ing models with billions of parameters (Agarwal et al.,

2023). To solve this problem, many efficient distillation approaches were proposed in recent years. They use knowl- edge caching (Jin et al.,2024;Zheng et al.,2021;Huang et al.,2024), self-distillation (Dong et al.,2023;Zhang et al.,2021;Naeem et al.,2025), dataset condensation (Zhao et al.,2020;Yin et al.,2024) and so on. However, these approaches focus on traditional downstream tasks such as computer vision or language processing, and few of them have delved deeply into efficient distillation using dataset se-

## 1. Introduction

Recent advancements in Large Language Models (LLMs) such as LLaMA and Qwen (Touvron et al.,2023;Bai et al., 2023;Yang et al.,2024a) have garnered significant atten- tion due to their remarkable capabilities and intelligence. However, these models impose substantial computational 1 State Key Laboratory of Complex & Critical Software Envi- ronment, Beihang University 2 School of Computer Science and Engineering, Beihang University 3 School of Artificial Intelligence, Beihang University 4 SenseTime Research 5 ETH Zurich. Corre- spondence to: Jinyang Guo *<*jinyangguo@buaa.edu.cn*>*.

*Proceedings of the 42* *nd* *International Conference on Machine* *Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

<!-- page 2 -->

<u>DA-KD: Difficulty-Aware Knowledge Distillation for Efficient Large Language Models</u>

lection for generative large models. As a result, it is still an open problem on how to effectively select informative sam- ples for distillation, which makes the efficient distillation for generative large models a non-trivial task.

To solve the aforementioned problems, in this paper, we pro- pose a knowledge distillation framework called **Difficulty-** **Aware Knowledge Distillation (DA-KD)**, which is de- signed for efficient distillation of large language models (LLMs). To achieve efficient distillation, we first construct a **Difficult-Aware Data Updating (DiffUp)** strategy, which includes a Distillation Difficulty Score (DDS) to measure the sample complexity based on the performance gap be- tween the teacher and student. As a result, we use DDS to filter out easy samples for condensed data volume. To strengthen data diversity and mitigate the forgetting problem when only difficult samples are adopt (Jiang et al.,2023), we also propose a Stratified Data Updating (SDU) strategy to improve the diversity of the dataset by gradually mixing samples with various DDS.

Furthermore, as we select difficult samples in the distillation process, we find the existing KD loss cannot provide robust optimization for the student. When extreme distribution occurs in the student, existing KD loss will encounter the explosion or vanishing gradient problem. Moreover, we also find it is challenging for existing KD loss to pay more at- tention to hard samples. To this end, we propose a new loss function called **Bidirectional Discrepancy Loss (BDL)** to restrict the gradient without explosion or vanishing. Specifi- cally, our BDL is built upon traditional KL divergence, but further incorporates combined probability distribution from both teacher and student. From our detailed analysis, we show that our BDL can not only effectively stabilize the optimization for the student but also enforce the distillation to pay more attention to the hard samples.

Our main contributions are summarized as follows:

▶ We propose difficulty-aware knowledge distillation (DA- KD) for efficient and effective LLM distillation, ▶ We also propose difficulty-aware data updating strategy to dynamically update the distillation dataset, which consists of a distillation difficult score for sample difficulty measure- ment and a stratified sampling strategy for data diversity. ▶ We propose a new loss function called bidirectional dis- crepancy loss, which can effectively stabilize the student training from difficult samples. ▶ Extensive experiments on multiple benchmark datasets demonstrate the effectiveness of our DA-KD framework.

## 2. Related Work

**Large language model.** Recent advancements in artifi- cial intelligence have garnered significant attention (Tao et al.,2022;2025b). In particular, Large language models

have emerged as transformative advancements in natural language processing, achieving state-of-the-art performance across a broad range of complex tasks. Prior works such as GPT (Radford,2018) introduced the use of stacked trans- former decoders, laying the foundation for subsequent inno- vations. Building on this, models like the Llama (Touvron et al.,2023;Dubey et al.,2024) and the Qwen (Yang et al., 2024a) series refined transformer architecture to improve efficiency and performance. Other approaches (Guo et al., 2023a;2021;2020) were also proposed in recent years by improving either training techniques or network architec- ture.

Despite these advancements, LLMs often require substantial computational resources due to their massive parameter sizes. In this work, we aim to train an efficient student model by using knowledge distillation technique.

**Knowledge distillation for LLM.** Knowledge distillation transfers knowledge from a large teacher model to a smaller student model (Hinton,2015) for faster inference. Recently, many knowledge distillation works were proposed for LLM. For example, MiniLLM (Gu et al.,2024) proposed reverse KL divergence (RKL) as the loss function to prevent the student from overestimating the low-probability regions of the teacher. GKD (Agarwal et al.,2023) additionally intro- duced generalized Jensen-Shannon divergence (JSD), and trained the student on its self-generated outputs to mitigate the training-inference mismatch. DistiLLM (Ko et al.,2024) proposed skew KL divergence (SKL) and skew reverse KL divergence (SRKL) that using a mixture of probability dis- tributions to improve the optimization stability.

However, current KD methods for LLM require a significant amount of training time (Xu et al.,2024;Ko et al.,2024), primarily due to the large-scale training data. In contrast, our DA-KD framework aims at not only effective but also efficient LLM knowledge distillation.

**Data selection for LLM training.** Data selection is increas- ingly recognized as crucial for optimizing the LLM fine- tuning process (Zhou et al.,2024;Wang et al.,2024a). These methods aim to prioritize subsets of data that are most valu- able for training, thus reducing computational costs while preserving or even improving model performance (Tao et al., 2025a). Several recent works have explored data selection for LLM fine-tuning. For example,Chen et al.(2023) in- troduced Alpagasus to directly leverage ChatGPT for rating each data sample, and select samples with higher ratings. Cao et al.(2023) proposed a linear rule-based approach named Instruction mining by training a linear function to identify the most effective data samples.Li et al.(2023) proposed the IFD method, where a LLM first learns from a small subset of data to acquire foundational capabilities and then uses this learned model to rate and select data from the original dataset.

<!-- page 3 -->

*Difficulty-aware Data Updating* *Bidirectional Discrepancy Distillation* *Stratified Data Updating* *Partition* 𝑟 Teacher Teacher ℒ 𝑥

| Initial Dataset | Teacher | ℒ!𝑥 | Partition | 𝑟 Stratified Data Updating |  | Teacher probability | 𝑝 | 1 − 𝜆 |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  | 𝒟$%& |  |  | 𝑃' |  |  |  |
| Current Dataset | 𝐷 |  | 𝒫 ∈ | 𝒫 ∋ | 𝒟 |  |  | 𝜆 | 𝐷 (𝑃, 𝑄) |  |  |
|  | Student | ℒ"!𝑥 Update | 𝒟 Mixing | 𝒟 𝜏 | Filter 𝒟#= 𝑟 | 𝐷 |  | Student probability | 𝑞( | 𝜆 1 − 𝜆𝑄' 𝑫 (𝒑, 𝒒) |  |

*power* 01/ / *power* 𝒟()*( 𝜏 +,- 1 − 𝜏 ()*( +,- !! !" # # .. ()*( +,-

*Backpropagation* 𝐁𝐃𝐋 𝜽

Overview of our Difficulty-Aware Knowledge Distillation framework.

*Figure 2.*

**Data selection for knowledge distillation.** Several studies also explored data selection methods in knowledge distilla- tion. For instance,Li et al.(2021) proposed an uncertainty- based data selection method for knowledge distillation of LLMs, whileLan et al.(2025) also introduced a data se- lection approach specifically designed for knowledge dis- tillation. However these methods are either solely based on student model (Li et al.,2021) or solely based on the teacher model (Lan et al.,2025), ignoring their discrepancy. Different from these works, we propose DA-KD framework to dynamically adjust distillation dataset in the distillation process based on the discrepancy between teacher and stu- dent.

## 3. Difficulty-Aware Knowledge Distillation

### 3.1. Overview

Figure2shows the overview of our DA-KD framework. Given a large teacher LLM and a small student LLM, we first calculate the cross-entropy loss of each sample in the initial dataset, based on which we calculate the distillation difficult score (DDS) for each sample. Then, we perform the stratified data updating based on the calculated DDS, in which the distillation dataset will be progressively reduced in this process. In the distillation procedure, we use our bidirectional discrepancy loss as the loss function, which can provide stable optimization and pay more attention to hard samples. We will introduce each component in our DA-KD framework in detail.

### 3.2. Difficulty-aware Data Updating

Knowledge distillation for LLMs often involves training a smaller student model to emulate the teacher model’s behavior. However, it is inefficient to use the entire dataset indiscriminately for distillation. To address this, we propose **Difficulty-aware data Updating (DiffUp)** to dynamically updates the distillation data based on the sample difficulty to reduce the computational costs of distillation.

3.2.1. DISTILLATION DIFFICULTY SCORE To evaluate the difficulty of sample *x* quantitatively, we first propose a new difficulty metric called **Distillation Difficulty** **Score (DDS)** as the ratio of the loss from student over that from teacher:
DDS(*x*) = <u>Lqθ(x)</u> *,* (1) *Lp*(*x*)

where *Lqθ*(*x*) and *Lp*(*x*) represent the cross-entropy loss between the sample *x* and the ground-truth from the student and teacher models, respectively.

This simple but effective DDS captures the distillation dif- ficulty based on the discrepancy between the teacher and student models. Figure3gives examples for different cases. When the student exhibits a high *Lqθ*(*x*) on a sample but the teacher has low loss *Lp*(*x*) on it, the DDS value becomes large. This situation indicates that the teacher is confident about the sample, whereas the student struggles to do so. Such case highlights the student model requires additional supervision from the teacher. Conversely, the DDS value remains small when both the student and teacher achieve low losses, which means the sample is well-learned by both models, making further distillation unnecessary. When both the teacher and student incur high losses, the DDS value remains small as well, which means the teacher’s under- standing of the sample is inadequate, limiting its ability to provide meaningful guidance. The essence of DDS is analogous to real-world teaching: effective teaching occurs when concentrating on difficult knowledge that the teacher understands well but the student finds challenging.

3.2.2. STRATIFIED DATA UPDATING In the distillation process, we propose **Stratified Data Up-** **dating** strategy to dynamically select the data in each epoch based on the aforementioned DDS. The core idea is to prior- itize difficult samples with high DDS value and gradually reduce the overall data volume in the next epoch for efficient distillation. Suppose we have a training dataset of size *N*, with the data selection ratio per epoch denoted by *r*. This im- plies that in each epoch, we actually select and use *rN* data samples for distillation. Initially, the selection ratio *r* is set to 1, meaning all available data is utilized at the beginning.

<!-- page 4 -->

**Outputs of Teacher and Student Loss DDS value** **Input**: How many cents do I have if I have 3 quarters? **Output**: If you have 3 quarters, you have 75 cents. 0.0003 **3685** **Output**: If you have 3 quarters, you have 30 cents. 1.1055

**Input**: Identify which animal species is alive or extinct: Megalania, Sea Turtle. **Output**: Sea Turtle is alive, Megalania is extinct. 0.0058

**0.9828**
**Output**: Sea Turtle is alive, Megalania is extinct. 0.0057

**Input**: What are the words of House Fowler? **Output**: Beware Our Talons 3.8324

**0.9729**
**Output**: First in Battle 3.7285

*Figure 3.* Examples of DDS values for different cases.

As distillation progresses, the selection ratio *r* gradually decreases following a linear or cosine decay schedule. If the total number of epochs is *E*, then the selection ratio *r* at the *e*-th epoch is:

<u>e</u> linear decay schedule : *r* = 1 *−,* *E* 1 *πe* 1

(2)
cosine decay schedule : *r* = cos + 2 *E* 2

This strategy allows the distillation process to focus on the most valuable data, thereby lowering training costs without sacrificing model performance.

**Stratified sampling for diversity.** Inspired byJiang et al. (2023), to strengthen data diversity and mitigate the forget- ting problem when only difficult samples are adopted, we further propose a stratified sampling strategy to mix various samples when updating distillation data. Specifically, at the beginning of each epoch, we first calculate the DDS for the whole dataset and sort them in descending order. Then, we split the dataset into two partitions by a number threshold *r*, creating low-DDS partition *Dlow*with 1 *− r* percentage of samples and high-DDS partition *Dhigh*with *r* percentage of samples, respectively. Then, we randomly draw sam- ples from both partitions to construct the data subset for distillation for the current epoch.

Formally, let us denote the whole dataset as set *D*. And the power set *P*(*D*) represents all subsets of *D*. It is obvious that *Dhigh, Dlow∈P*(*D*) and *Dhigh∪Dlow*= *D*. Next, *P* *high* 1*−τ* represents the collection of subsets in the power set of *Dhigh*that contain (1 *− τ*)*|Dhigh|* elements, i.e.,

*∀S ∈ P* *high* 1*−τ* *, |S|* = (1 *− τ*)*|Dhigh|,* and *S ⊆Dhigh,* (3)

where *|·|* returns the number of element of the set, and *τ* is to balance the selection ratio in high and low partitions. Before the start of each epoch, we compute and sort to obtain *′* *Dhigh*, and then randomly select an element *Dhigh*from *−τ′−τ′* *P* *high*, i.e., *Dhigh∈RP* *high* and *Dhigh⊆ Dhigh*, where

*∈R*denotes a random selection from a finite set. The same *′* procedure is applied to the *Dlow*set, where an element *Dlow* *τ* *′* *τ* is randomly selected from *Plow*, i.e., *Dlow∈RPlow*, where *P* *τ* represents the collection of subsets in the power set *low* of *Dlow*that contain *τ |Dhigh|* elements. Thus, the activated data for this training epoch is *D* *′* = *D* *′* *∪D* *′*, and *′ ′* *low high* is easy to know that *|Dlow|* + *|Dhigh|* = *τ |Dhigh|* + (1 *−* *τ*)*|Dhigh|* = *r|D|*.

This stratified sampling strategy ensures the distillation dataset not only prioritizes challenging samples but also retains representative easier examples, which prevents poten- tial biases toward high-DDS samples and allows the student to effectively generalize across the entire data distribution.

### 3.3. Bidirectional Discrepancy Distillation

As we use our DiffUp approach to construct our distilla- tion dataset, the retained training samples are often more challenging, exhibiting greater discrepancies between the teacher and student.

To achieve robust and stable distillation, we propose a new distillation loss function called **Bidirectional Discrepancy** **Loss (BDL)** building upon KL divergence, but further inte- grate the two measured probability distributions, which can be formally expressed as:

*D*BDL(*p,qθ*) =

(4)
*D*KL(((1 *− λ*)*p* + *λqθ*)*∥*(*λp* + (1 *− λ*)*qθ*))*,*

where *p* and *qθ*denote the probability distributions from teacher and student, respectively. *λ* is the coefficient to balance the contribution of *p* and *qθ*during the mixing. *D*KL denotes the standard KL divergence.

**Remark.** Our BDL can perform well on difficult samples because it can provide robust optimization for the student. Specifically, let us denote *Pm*= (1 *− λ*)*p* +*λqθ*and *Qm*= *λp* + (1 *− λ*)*qθ*, we compute the gradient of BDL w.r.t. student parameter *θ*:

*∇θD*BDL(*p,qθ*) = *∇θD*KL(*Pm,Qm*) X <u>P</u> <u>m(x)</u> = *∇θPm*(*x*) log *Qm*(*x*) *x* X <u>P</u> <u>m(x) Pm(x)</u> = [*λ*log + *λ −* (1 *− λ*)] *∇θqθ*(*x*)*,* *x*| <u>Qm(x)</u> {z <u>Qm(x)</u> } *C*(*x*)

(5)
where *C*(*x*) determines the direction and magnitude of each term in *∇θqθ*(*x*). Next, we substitute *Pm*and *Qm*into *Pm*(*x*)*/Qm*(*x*) to analyze the influence of the blend distri- bution on the gradient:

<u>Pm(1 − λ)p + λqθ</u> *Q* = *λp* + (1 *− λ*)*q*

*.* (6)
*m θ*

<!-- page 5 -->

Assuming that the distribution of *qθ*may be extremely small

| or large for hard samples. If q |  | (x) → 0, we have |  | → |
| --- | --- | --- | --- | --- |
|  |  | θ |  | Q x)) |
| (1−λ)p(x) | 1−λ |  |  |  |
| λp(x) | λ |  | θ |  |
| P (x) | λq (x) | λ |  |  |
| Q (x) | (1−λ)q (x) | 1−λ |  |  |

*θ Q* <u>Pm</u> *m* <u>(</u> ( <u>x</u> *x* <u>)</u> ) =*.* On the contrary, if *q* (*x*) *→∞*, we have *m* *→* *θ* =*.* A similar phenomenon can *m θ* be observed when *p*(*x*) approaches 0 or *∞*.

This leads to our first conclusion: BDL stabilizes the train- ing by restricting the gradients without explosion or van- ishing. Specifically, the range of *C*(*x*) is solely determined by *λ*, and is independent of *qθ*or *p*. *C*(*x*) falls between (*λ*log 1*−* *λλ* + 2 *−* *λ* 1 and *λ*log 1*−* *λλ* ) (see Figure5in the ap- pendix for illustration). Therefore, since *λ* is a pre-defined value, the range of *C*(*x*) is finite and known in advance. This prevents *C*(*x*) from becoming excessively large or small due to *qθ*or *p* approaching extreme or less active, thereby avoiding gradient explosion or vanishing.

Furthermore, by analyzing the range and monotonicity of *C*(*x*) in AppendixA, we derive our second conclusion: BDL emphasizes the samples that have distinct output proba- bilities by teacher and student model (i.e., difficult samples). Specifically, denoting *z* = *Pm/Qm*, then we have

<u>∂C(x) λ</u> = *−* (1 *− λ*)*.* (7) *∂z z*

when *z <* 1*−* <u>λ</u> *λ*, <u>∂C</u> *∂z*

<u>(x)</u> exhibit a positive value, indicating
*C*(*x*) is monotonically increasing with *z* = *Pm/Qm*. From our experiments, *λ* = 0*.*9 excels the best performance. Con- sidering the distribution from both teacher and student are usually small, *z &lt;* 1*−*

<u>0.</u> 0 <u>9</u>
*.*9 takes most of cases in the distil-
lation. Therefore, *C*(*x*) can be treated as monotonically increasing with *z*. From AppendixA, we also demonstrate *z* = *Pm/Qm*is also monotonically increasing with *qθ/p* when *λ &gt;* 0*.*5. As a result, *C*(*x*) is monotonically increas- ing with respect to *qθ/p* in our setting. We empirically observe most difficult samples have *qθ≫ p*, so we have larger *C*(*x*) on these difficult samples. Although this as- sumption does not always hold, we find it is helpful for the distillation of our DA-KD. This property can enlarge the gradient of these hard samples, which enforce our BDL to pay more attention on them.

|  | P |  |
| --- | --- | --- |
|  | Q | qp |
|  |  | P |
| q |  | Q |
| p |  | θ |

Meanwhile, when *λ ∈* (0*,*0*.*5), <u>m</u> *m* is closer to *θ*, and *C*(*x*) is constantly negative, making BDL behaves similar <u>m</u> to standard KL divergence. When *λ >* 0*.*5, *m* transits <u>θ</u> to, and *C*(*x*) takes positive values to make *q* increase, making BDL behaves similar to reverse KL divergence. However, *λ* = 0*.*9 excels the best performance when *C*(*x*) is either positive or negative, which provides a larger range <u>Pm</u> of *C*(*x*), and also brings a bigger *Qm* (see AppendixA). This demonstrates that the combination of KL and reverse KL helps the training and convergence of the student. But a higher *λ* may also lead to aggressiveness and gradient explosion. The complete derivation and proof can be found

<u>Algorithm 1 Difficulty-aware data updating in our DA-KD</u> **Input:** Initial selection ratio *r* = 1; the whole dataset *D* with size *N* = *|D|*; balancing coefficient *τ*; total train- ing epochs *E*; teacher and student model *θ* and *θ*. *T S* **Output:** Trained student model *θS*. 1: *Iterate the model using D, and obtain initial DDS* 2: **for** *i ←* 2 to *E* **do** 3: *Update the selection ratio r based on Eq. (2)* 4: *Lqθ*(*D*)*, L* <u>L</u> *p* ( <u>(</u> *D* <u>D</u> ) <u>)</u> *← θT*(*D*)*,θS*(*D*) 5: DDS *←* <u>q</u> <u>θ</u> *Lp* (*D*) 6: *Dord← Descending ordered D with updated DDS* 7: *Dhigh←Dord*[: *rN*] *Dlow←Dord*[*rN* :] 8: *Plow* *τ* *←∀t ∈P*(*Dlow*)*,s.t.|t|* = *τ |Dhigh|* *P* *high* 1*−τ* *←∀t ∈P*(*Dhigh*)*,s.t.|t|* = (1 *− τ*)*|Dhigh|*

9: *D* *′* *, D* *′ ∈* *←P* *Rτ* *, P¹* *−τ* *low high low high* *′* *′ ′* 10: *D ←Dhigh*+ *Dlow* 11: *yT,yS← θT*(*D* *′* )*,θS*(*D* *′* ) 12: *L*BDL*← D*BDL(*yT,yS*) 13: *L*BDL.backward() 14: **end for** 15: **return** Student model *θS*

in AppendixA.

In a word, by gradient analysis, we claim that BDL not only stabilizes the optimization process inherently, but also emphasizes the hard samples that induce distinct activations. Compared to the traditional KL divergence, the blending of teacher and student distributions in the numerator and denominator results in smoother gradients, avoiding unsta- ble updates in extreme cases where the teacher and student distributions differ significantly. At the same time, the gra- dients are regularized based on both distributions, empha- sizing the importance of hard samples during training. We provide a detailed parameter analysis experiment to verify the behavior of BDL under different *λ* to show that our BDL ensures more stable and efficient update during training.

After using the proposed BDL and the DiffUp methods, we can effectively focus on more difficult but informative samples, and thus achieve efficient distillation. To provide better explanation of the process for our DA-KD framework, we present a description of the procedure in Algorithm1.

## 4. Experiments

### 4.1. Experiments Setup

**Tasks and Datasets.** To validate the effectiveness of our DA-KD framework, we conduct two categories of experi- ments: task-agnostic instruction following and task-specific experiments.

<!-- page 6 -->

*Table 1.* Results of task-agnostic instruction following on Llama2 and Qwen2.5 models. We report the average ROUGE-L scores across

five random seeds. The **bold** <u>and underline</u> indicate the best and second-best results, respectively.

**Model**

| #Params | Method | DollyEval | SelfInst | Super-Natural | Unnatural | VicunaEval | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | Teacher | 29.61 | 20.98 | 34.11 | 35.56 | 19.19 | 27.89 |
|  | SFT | 27.52 | 19.48 | 30.93 | 32.09 | 18.24 | 25.65 |
|  | KD-KL (Hinton,2015) | 27.71 | 19.19 | 30.24 | 31.75 | 17.73 | 25.32 |
|  | KD-RKL (Gu et al.,2024) | 28.60 | 19.74 | 35.18 | 36.26 | 18.71 | 27.70 |
| 2.7B | SeqKD (Kim & Rush,2016) | 27.78 | 18.64 | 30.95 | 32.14 | 18.13 | 25.53 |
|  | GKD (Agarwal et al.,2023) | 28.72 | 19.48 | 33.12 | 33.74 | 19.17 | 26.85 |
|  | Distillm (Ko et al.,2024) | 27.93 | 18.90 | 32.03 | 35.42 | 18.76 | 26.61 |
|  | DA-KD (ours) | 29.04 | 19.50 | 35.29 | 37.39 | 19.34 | 28.11 |
|  | SFT | 25.85 | 14.59 | 24.13 | 28.22 | 17.41 | 22.04 |
|  | KD-KL (Hinton,2015) | 26.17 | 15.13 | 24.97 | 29.22 | 17.34 | 22.57 |
|  | KD-RKL (Gu et al.,2024) | 25.66 | 15.01 | 27.89 | 32.39 | 18.34 | 23.86 |
| 1.3B | SeqKD (Kim & Rush,2016) | 25.98 | 15.00 | 25.36 | 29.83 | 17.66 | 22.76 |
|  | GKD (Agarwal et al.,2023) | 26.55 | 16.51 | 27.88 | 30.63 | 17.66 | 23.85 |
|  | Distillm (Ko et al.,2024) | 27.16 | 16.52 | 27.27 | 31.98 | 17.70 | 24.13 |
|  | DA-KD (ours) | 26.75 | 18.05 | 32.35 | 36.09 | 17.53 | 26.15 |
| 7B | Teacher | 31.37 | 25.98 | 43.61 | 40.19 | 23.15 | 32.86 |
|  | SFT | 28.05 | 21.12 | 34.21 | 37.06 | 19.22 | 27.93 |
|  | KD-KL (Hinton,2015) | 28.01 | 21.97 | 37.24 | 37.94 | 19.98 | 29.03 |
|  | KD-RKL (Gu et al.,2024) | 28.40 | 20.60 | 37.93 | 39.46 | 21.18 | 29.51 |
| 1.5B | SeqKD (Kim & Rush,2016) | 28.21 | 20.98 | 35.79 | 36.52 | 21.46 | 28.59 |
|  | GKD (Agarwal et al.,2023) | 28.89 | 23.78 | 44.76 | 39.56 | 23.80 | 32.16 |
|  | Distillm (Ko et al.,2024) | 29.38 | 21.74 | 40.55 | 37.41 | 22.96 | 30.41 |
|  | DA-KD (ours) | 28.91 | 24.08 | 45.73 | 42.31 | 24.04 | 33.01 |
|  | SFT | 25.98 | 14.89 | 30.74 | 32.82 | 16.54 | 24.19 |
|  | KD-KL (Hinton,2015) | 24.59 | 16.07 | 30.00 | 30.95 | 17.57 | 23.84 |
|  | KD-RKL (Gu et al.,2024) | 25.09 | 15.30 | 29.06 | 32.01 | 19.08 | 24.11 |
| 0.5B | SeqKD (Kim & Rush,2016) | 26.28 | 17.40 | 29.57 | 31.64 | 18.90 | 24.76 |
|  | GKD (Agarwal et al.,2023) | 26.24 | 18.74 | 38.45 | 33.31 | 21.56 | 27.66 |
|  | Distillm (Ko et al.,2024) | 27.72 | 19.55 | 42.19 | 37.80 | 23.94 | 30.24 |
|  | DA-KD (ours) | 27.36 | 19.74 | 43.63 | 39.52 | 21.97 | 30.44 |

**Llama2**

**Qwen2.5**

For instruction following experiments, we choose the databricks-dolly (Conover et al.,2023) dataset pro- cessed byGu et al.(2024) for distillation. Then, we evaluate the trained student models on five instruction-following datasets: Dolly evaluation (Conover et al.,2023), Self- Instruct (Wang et al.,2022a), Super-Natural Instructions (Wang et al.,2022b), Unnatural Instruction(Honovich et al.,

2022. and Vicuna evaluation(Chiang et al.,2023). The eval- uation metric is ROUGE-L (Lin,2004). To mitigate the randomness, we report the average ROUGE-L score under five different random seeds. For task-specific experiments, we consider two distinct tasks for evaluation: text summarization using SAM- Sum (Gliwa et al.,2019), and mathematical reasoning with GSM8K (Cobbe et al.,2021). We use ROUGE-L score and zero-shot accuracy for measuring, respectively. **Models.** We evaluate our DA-KD using multiple teacher- student model pairs. For task-agnostic instruction fol-

lowing, we employ LLaMA2-7B (Touvron et al.,2023) and Qwen2.5-7B (Yang et al.,2024a) as teacher models, with Sheared-LLaMA2-2.7B/1.3B (Xia et al.,2023) and Qwen2.5-1.5B/0.5B as their respective students. For task- specific experiments, we utilize LLaMA2-7B and Qwen2.5- 7B as teachers, distilling knowledge into LLaMA2-1.3B and Qwen2.5-1.5B, respectively. Additionally, we include LLaMA3.2-3B (Dubey et al.,2024) as the teacher model with LLaMA3.2-1B as the student.

**Implementation details.** We train all models for 10 epochs using a batch size of 8. We use the AdamW optimizer and a cosine learning rate scheduler. The initial learning rate is set as 1e-5. In our implementation, we set *τ* and *λ* as 0.1 and 0.9, respectively. We use a cosine decay schedule to gradually reduce the data selection ratio *r*.

**Comparison Methods.** We compare our approach with six methods: SFT, KD-KL (Hinton,2015), KD-RKL (Gu et al.,

2024), SeqKD (Kim & Rush,2016), GKD (Agarwal et al.,

<!-- page 7 -->

2023), Distillm (Ko et al.,2024). SFT means we directly fine-tune the student without distillation.

### 4.2. Main Results

**Task-Agnostic Instruction-Following.** Table1presents the results of task-agnostic instruction-following experiments. Our proposed DA-KD method achieves the highest ROUGE- L scores on most of the evaluation datasets, consistently outperforming other methods across different model types and sizes. For instance, in the case of the Llama2-2.7B model, our approach achieves an average ROUGE-L score of 28.11, surpassing the state-of-the-art methods KD-RKL (27.70) and GKD (26.85). Even on the smallest Qwen2.5-

0.5B model, our method attains an average ROUGE-L score of 30.44, exceeding Distillm (30.24) and all other baselines. Notably, both Llama2-2.7B and Qwen2.5-1.5B outperform their respective teacher models in terms of average perfor- mance. We hypothesize that this improvement may come from the more informative training datasets brought by our DiffUP approach and the more stable optimization process caused by BDL. Moreover, we successfully compress the Qwen2.5 model from 7B to 0.5B (14*×* reduction in disk storage) with only a 2.42 drop in average performance. Furthermore, com- pressing Qwen2.5 from 7B to 1.5B (4.7*×* storage reduction) not only preserves performance but even yields a slight improvement. Similar results are observed in the Llama2

| Method | Iterations | Traing time (minutes) |
| --- | --- | --- |
| SFT | 3570 | 62.81 |
| KD-KL (Hinton,2015) | 3570 | 140.75 |
| KD-RKL (Gu et al.,2024) | 3570 | 141.40 |
| SeqKD (Kim & Rush,2016) | 3570 | 159.73 |
| GKD (Agarwal et al.,2023) | 3570 | 408.24 |
| Distillm (Ko et al.,2024) | 3570 | 213.34 |
| DA-KD (ours) | 1963 | 106.35 |

 experiments. It further demonstrates the effectiveness of our DA-KD framework, highlighting the potential of our proposed distillation framework in producing smaller large models with minor computational resources and time costs. **Task-Specific Experiments.** Table2reports the results of task-specific experiments conducted on the Llama2-1.3B, Qwen2.5-1.5B and Llama3.2-1B for text summarization and mathematical reasoning. Our DA-KD method achieves the best performance across both tasks, demonstrating its effectiveness in diverse task-specific scenarios. Specifically, for the text summarization task on the SAMSum dataset, our approach consistently surpasses previous state-of-the-art methods. For example, on Llama2-1.3B and Llama3.2-1B, our DA-KD achieves 39.20 and 32.92 respectively, outper- forming Distillm (38.73 and 32.53). Notably, on Qwen2.5, the student model distilled by our DA-KD achieves a score of 40.05, surpassing even the 7B teacher model. The re- markable result underscores DA-KD’s ability to empower a lightweight student model to retain and even enhance high-quality text generation capabilities. For the mathematical reasoning task on the GSM8K dataset, our method achieves a zero-shot accuracy of 54.66% on Qwen2.5, 14.56% on Llama2, and 22.37% on Llama3.2, demonstrating significant improvements over competitors.
*Table 2.* Results of task-specific experiments on Llama2-1.3B,

Qwen2.5-1.5B and Llama3.2-1B. The **bold** and <u>underline</u> mark- ings indicate the best and second-best results, respectively.

**Model Method SAMSum GSM8K** Teacher (7B) 40.84 38.67 SFT 37.11 11.52

Llama2 KD-KL (Hinton,2015) 37.80 11.37 KD-RKL (Gu et al.,2024) 38.45 9.40 GKD (Agarwal et al.,2023) 38.39 14.18 Distillm (Ko et al.,2024) 38.73 12.59 DA-KD (ours) **39.20 14.56** Teacher (7B) 39.70 71.72 SFT 37.74 41.77 KD-KL (Hinton,2015) 37.91 44.50 Qwen2.5 KD-RKL (Gu et al.,2024) 38.75 44.88 GKD (Agarwal et al.,2023) 38.89 50.80 Distillm (Ko et al.,2024) 39.21 46.70 DA-KD (ours) **40.05 54.66** Teacher (3B) 33.47 37.83 SFT 31.78 12.59 KD-KL (Hinton,2015) 29.57 7.28 Llama3.2 KD-RKL (Gu et al.,2024) 30.51 14.03 GKD (Agarwal et al.,2023) 30.43 17.82 Distillm (Ko et al.,2024) 32.53 20.17 DA-KD (ours) **32.92 22.37**

*Table 3.* Training efficiency comparison.

However, performance degradation is still observed for smaller models on GSM8K. One possible explanation is the inherent complexity of mathematical reasoning, requiring stronger logical inference and multi-step reasoning capabili- ties. Consequently, distilling such knowledge into a smaller model is inherently more challenging, and remains an open problem for future work to enhance knowledge transfer in mathematical reasoning.

**Training costs.** As our objective is to develop an effective and efficient distillation framework for large language mod- els, we also evaluate the training efficiency of our method compared to existing work. Table3presents the number of training iterations and time required for end-to-end distil- lation. All the test cases compress Llama2-7B into a 2.7B model using four NVIDIA A800 GPUs.

<!-- page 8 -->

*Table 4.* Ablation study of our DiffUp method.

*Table 5.* Comparison results for different distillation loss functions

combined with DiffUP. We report the average ROUGE-L scores

| Method | Dolly | SelfInst | Super-N | Unnatural | Vicuna | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| DiffUp | 26.75 | 18.05 | 32.35 | 36.09 | 17.53 | 26.15 |
| w/o DDS | 26.05 | 16.70 | 31.27 | 33.86 | 17.53 | 25.08 |
| w/o SDU | 26.65 | 17.37 | 31.11 | 34.56 | 16.54 | 25.24 |
| w/o DiffUp | 26.73 | 18.51 | 32.07 | 35.64 | 16.58 | 25.91 |

across five random seeds. The **bold** and <u>underline</u> markings indi- cate the best and second-best results, respectively.

| Loss | Dolly | SelfInst | Super-N | Unnatural | Vicuna | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| KL | 23.69 | 14.40 | 24.97 | 26.16 | 15.98 | 21.04 |
| RKL | 25.35 | 14.72 | 27.31 | 32.15 | 18.18 | 23.54 |
| JSD | 24.80 | 16.86 | 31.18 | 31.70 | 16.79 | 24.26 |
| SKL | 26.02 | 17.31 | 30.76 | 33.02 | 17.72 | 24.96 |
| SRKL | 26.43 | 16.91 | 29.94 | 34.56 | 17.12 | 24.99 |
| BDL | 26.75 | 18.05 | 32.35 | 36.09 | 17.53 | 26.15 |

We take SFT as the baseline performance since it directly fine-tunes the student model on the corresponding dataset without knowledge distillation. Compared to all the other typical and commonly-used distillation frameworks, our DA-KD significantly reduces the computational cost of dis- tillation while maintaining superior performance. Specifi- cally, DA-KD requires only 1,963 training iterations, which is 55% fewer compared to all other distillation methods.

In terms of training time, our DA-KD method requires only

26.1% of the GPU hours compared to GKD (106.35 vs.
408.24 minutes) and 49.9% compared to Distillm (106.35 vs. 213.34 minutes), demonstrating remarkable efficiency compared to other methods. That is because as the train- ing process goes on, we use fewer data with hard samples while filtering out the most simple ones, which brings sig- nificant time saving for each single epoch. Methods such as GKD and Distillm incur substantial computational costs because they rely on student-generated outputs during train- ing, which increases processing overhead. Although it will introduce an extra computational cost when computing DDS in DA-KD (for example, the total time of 106.35 minutes in Table3consists of 28.31 minutes of DDS computation and 78.04 minutes of distillation), we can achieve a reduc- tion of 55% in training iterations by using DDS to filter the distillation data. As a result, we can still achieve an overall reduction in computational cost compared to existing KD methods.

### 4.3. Ablation Study

In this section, we distill Llama2-7B to 1.3B as an exam- ple and conduct extensive ablation studies to validate the effectiveness of each component of our DA-KD approach.

4.3.1. EFFECT OF DIFFUP To verify the effectiveness of our proposed difficulty metric DDS, we first remove each component in DiffUp. The result denoted as “**w/o DDS**” in Table4means we randomly select data for distillation at each epoch instead of according to their DDS, showing that our DDS metric can bring 1.07 improvement on the average ROUGE-L score, highlighting the importance of DDS in ordering and selecting challeng- ing samples for distillation. For comparison denoted as “**w/o SDU**”, we remove the mixing strategy, making all the samples in the distillation dataset being sampled from the high-DDS partition. So SDU brings 0.91 improvement on the average ROUGE-L score. Finally, when we remove the
Unnatural Avg Std.. Super-N

Dolly

Vicuna SelfInst

*Figure 4.* Hyperparameter analysis of *λ* in BDL.

whole DiffUp method and use the entire dataset during dis- tillation process (“**w/o DiffUp**”), the performance degrades

0.24 compared to ours, indicating that dynamically updating the dataset based on the complexity of samples improves the model’s performance.
4.3.2. EFFECT OF BDL **Comparisons to Other Distillation Losses.** To verify the effectiveness of our proposed BDL, we conduct exper- iments by replacing BDL with other objective functions: KL (Hinton,2015), KL (Gu et al.,2024), JSD (Agarwal et al.,2023), SKL (Ko et al.,2024), and SRKL (Ko et al.,
2024). As shown in Table5, BDL achieves the highest av- erage ROUGE-L score of 26.15, surpassing SRKL by 1.16, and has even larger margin compared to other loss functions, which demonstrates our BDL can stabilize the training and also learn more information in difficult samples. **Parameter Analysis on** *λ***.** We conduct the experiment to investigate the performance with different values of *λ* in BDL, and the result is shown in Figure4, which is con- sistent to our derivation in AppendixA. When *λ* = 0*.*5, the gradients will have relatively small coefficients which lead to gradient vanishing and performance degradation. When *λ ∈* (0*.*1*,*0*.*5), BDL behaves similar to standard KL divergence and constantly improves the model performance. Moreover, when *λ ∈* (0*.*6*,*0*.*9), it consistently leads to per- formance gains, and reaches its optimal when *λ* = 0*.*9, which demonstrates that the combination of KL and reverse KL helps the model to converge. The results in Figure4con- firm the conclusions of our theoretical derivation. However,

<!-- page 9 -->

determining the optimal *λ*, as well as fine-grained tuning of *λ* for the best efficiency, remains an open problem.

## 5. Conclusion

In this paper, we introduce Difficulty-Aware Knowledge Distillation (DA-KD), a novel framework designed to en- hance the efficiency and effectiveness of LLM distillation. It includes Difficulty-Aware Data Updating (DiffUp) to filters easy samples using a Distillation Difficulty Score (DDS), and Stratified Data Updating (SDU) to mitigates catastrophic forgetting by mixing samples from all DDS partitions. Additionally, we propose Bidirectional Discrep- ancy Loss (BDL) to stabilize student training by preventing gradient explosion or vanishing when distilling difficult samples. Extensive experiments demonstrate that DA-KD outperforms state-of-the-art knowledge distillation methods and even surpasses the teacher model in several datasets with only half training cost, making it a highly effective solution for efficient LLM distillation.

One of the limitations of our DA-KD is that its performance depends on the teacher model quality. If the teacher model provides incorrect prediction, the DDS mechanism may misidentify difficult samples, potentially affecting student performance.

## Acknowledgements

This work was supported by the National Science and Tech- nology Major Project (2022ZD0116405), Beijing Municipal Science and Technology Project (No. Z231100010323002), National Natural Science Foundation of China (Nos. 62306025, 92367204, 62476018).

## Impact Statement

This paper presents work whose goal is to advance the field of LLM knowledge distillation. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Agarwal, R., Vieillard, N., Stanczyk, P., Ramos, S., Geist,

M., and Bachem, O. Gkd: Generalized knowledge distilla- tion for auto-regressive sequence models. *arXiv preprint* *arXiv:2306.13649*, 2023.
Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan,

Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023.
Cao, Y., Kang, Y., Wang, C., and Sun, L. Instruction min-

ing: Instruction data selection for tuning large language models. *arXiv preprint arXiv:2307.06290*, 2023.

Chen, L., Li, S., Yan, J., Wang, H., Gunaratna, K., Yadav,

V., Tang, Z., Srinivasan, V., Zhou, T., Huang, H., et al. Alpagasus: Training a better alpaca with fewer data. *arXiv* *preprint arXiv:2307.08701*, 2023.
Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang,

H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. *See [https://vicuna](https://vicuna). lmsys. org* *(accessed 14 April 2023)*, 2(3):6, 2023.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,

R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*,
2021.
Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah,

S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R. Free dolly: Introducing the world’s first truly open instruction- tuned llm. *Company Blog of Databricks*, 2023.
Dong, X., Bao, J., Zheng, Y., Zhang, T., Chen, D., Yang, H., Zeng, M., Zhang, W., Yuan, L., Chen, D., et al. Maskclip: Masked self-distillation advances contrastive language- image pretraining. In *Proceedings of the IEEE/CVF Con-* *ference on Computer Vision and Pattern Recognition*, pp. 10995–11005, 2023.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,

A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan,
A., et al. The llama 3 herd of models. *arXiv preprint* *arXiv:2407.21783*, 2024.
Gliwa, B., Mochol, I., Biesek, M., and Wawer, A. Samsum corpus: A human-annotated dialogue dataset for abstrac- tive summarization. *arXiv preprint arXiv:1911.12237*,

2019.
Gu, Y., Dong, L., Wei, F., and Huang, M. Minillm: Knowl- edge distillation of large language models. In *The Twelfth* *International Conference on Learning Representations*,

2024.
Guo, J., Ouyang, W., and Xu, D. Multi-dimensional pruning: A unified framework for model compression. In *CVPR*,

2020.
Guo, J., Liu, J., and Xu, D. Jointpruning: Pruning networks along multiple dimensions for efficient point cloud pro- cessing. *IEEE Transactions on Circuits and Systems for* *Video Technology*, 2021.

Guo, J., Liu, J., and Xu, D. 3d-pruning: A model com- pression framework for efficient 3d action recognition.

<!-- page 10 -->

*IEEE Transactions on Circuits and Systems for Video* *Technology*, 32(12):8717–8729, 2022.

Guo, J., Xu, D., and Lu, G. Cbanet: Towards complexity and bitrate adaptive deep image compression using a single network. *IEEE Transactions on Image Processing*, 2023a.

Guo, J., Xu, D., and Ouyang, W. Multidimensional pruning and its extension: A unified framework for model com- pression. *IEEE Transactions on Neural Networks and* *Learning Systems*, 2023b.

Guo, J., Wu, J., Wang, Z., Liu, J., Yang, G., Ding, Y., Gong,

R., Qin, H., and Liu, X. Compressing large language mod- els by joint sparsification and quantization. In *Forty-first* *International Conference on Machine Learning*, 2024.
Hinton, G. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.

Honovich, O., Scialom, T., Levy, O., and Schick, T. Unnat- ural instructions: Tuning language models with (almost) no human labor. *arXiv preprint arXiv:2212.09689*, 2022.

Huang, Y., Wang, Z., Gong, R., Liu, J., Zhang, X., and Zhang, J. Harmonica: Harmonizing training and infer- ence for better feature cache in diffusion transformer acceleration. *arXiv preprint arXiv:2410.01723*, 2024.

Jiang, Y., Chan, C., Chen, M., and Wang, W. Lion: Adver- sarial distillation of proprietary large language models. In *Proceedings of the 2023 Conference on Empirical Meth-* *ods in Natural Language Processing*, pp. 3134–3154,

2023.
Jin, C., Zhang, Z., Jiang, X., Liu, F., Liu, X., Liu, X., and Jin,

X. Ragcache: Efficient knowledge caching for retrieval- augmented generation. *arXiv preprint arXiv:2404.12457*,
2024.
Kim, Y. and Rush, A. M. Sequence-level knowledge distil- lation. *arXiv preprint arXiv:1606.07947*, 2016.

Ko, J., Kim, S., Chen, T., and Yun, S.-Y. Distillm: Towards streamlined distillation for large language models. *arXiv* *preprint arXiv:2402.03898*, 2024.

Lan, W., Cheung, Y.-m., Xu, Q., Liu, B., Hu, Z., Li, M., and Chen, Z. Improve knowledge distillation via label revi- sion and data selection. *IEEE Transactions on Cognitive* *and Developmental Systems*, 2025.

Li, L., Lin, Y., Ren, S., Li, P., Zhou, J., and Sun, X. Dynamic knowledge distillation for pre-trained language models. In *Proceedings of the 2021 Conference on Empirical* *Methods in Natural Language Processing*, pp. 379–389,

2021.
Li, M., Zhang, Y., Li, Z., Chen, J., Chen, L., Cheng, N., Wang, J., Zhou, T., and Xiao, J. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. *arXiv preprint arXiv:2308.12032*,

2023.
Lin, C.-Y. Rouge: A package for automatic evaluation of summaries. In *Text summarization branches out*, pp. 74–81, 2004.

Lv, C., Chen, H., Guo, J., Ding, Y., and Liu, X. Ptq4sam: Post-training quantization for segment anything. In *Pro-* *ceedings of the IEEE/CVF Conference on Computer Vi-* *sion and Pattern Recognition*, pp. 15941–15951, 2024.

Naeem, M. F., Xian, Y., Zhai, X., Hoyer, L., Van Gool,

L., and Tombari, F. Silc: Improving vision language pretraining with self-distillation. In *European Conference* *on Computer Vision*, pp. 38–55. Springer, 2025.
Radford, A. Improving language understanding by genera- tive pre-training. 2018.

Tao, R., Wang, T., Wu, Z., Liu, C., Liu, A., and Liu, X. Few-shot x-ray prohibited item detection: A benchmark and weak-feature enhancement network. In *Proceedings* *of the 30th ACM international conference on multimedia*, pp. 2012–2020, 2022.

Tao, R., Le, M., Tan, C., Liu, H., Qin, H., and Zhao, Y. Oddn: Addressing unpaired data challenges in open-world deep- fake detection on online social networks. In *Proceedings* *of the AAAI Conference on Artificial Intelligence*, vol- ume 39, pp. 799–807, 2025a.

Tao, R., Wang, H., Guo, Y., Chen, H., Zhang, L., Liu, X., Wei, Y., and Zhao, Y. Dual-view x-ray detection: Can ai detect prohibited items from dual-view x-ray images like humans? In *2025 IEEE/CVF Conference on Computer* *Vision and Pattern Recognition (CVPR)*. IEEE, 2025b.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,

A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine- tuned chat models. *arXiv preprint arXiv:2307.09288*,
2023.
Wang, J., Zhang, B., Du, Q., Zhang, J., and Chu, D. A survey on data selection for llm instruction tuning. *arXiv* *preprint arXiv:2402.05123*, 2024a.

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language models with self-generated instructions. *arXiv* *preprint arXiv:2212.10560*, 2022a.

Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran,

<!-- page 11 -->

A. S., Naik, A., Stap, D., et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. *arXiv preprint arXiv:2204.07705*, 2022b.
Wang, Z., Guo, J., Gong, R., Yong, Y., Liu, A., Huang,

Y., Liu, J., and Liu, X. Ptsbench: A comprehensive post-training sparsity benchmark towards algorithms and models. In *ACM Multimedia 2024*, 2024b.
Xia, M., Gao, T., Zeng, Z., and Chen, D. Sheared llama: Accelerating language model pre-training via structured pruning. *arXiv preprint arXiv:2310.06694*, 2023.

Xu, X., Li, M., Tao, C., Shen, T., Cheng, R., Li, J., Xu,

C., Tao, D., and Zhou, T. A survey on knowledge distillation of large language models. *arXiv preprint* *arXiv:2402.13116*, 2024.
Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li,

C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. *arXiv preprint arXiv:2412.15115*, 2024a.
Yang, G., He, C., Guo, J., Wu, J., Ding, Y., Liu, A., Qin,

H., Ji, P., and Liu, X. Llmcbench: Benchmarking large language model compression for efficient deployment. *NeurIPS*, 2024b.
Yin, Z., Xing, E., and Shen, Z. Squeeze, recover and rela- bel: Dataset condensation at imagenet scale from a new perspective. *Advances in Neural Information Processing* *Systems*, 36, 2024.

Zhang, L., Bao, C., and Ma, K. Self-distillation: Towards efficient and compact neural networks. *IEEE Transac-* *tions on Pattern Analysis and Machine Intelligence*, 44

(8):4388–4403, 2021.
Zhao, B., Mopuri, K. R., and Bilen, H. Dataset condensation with gradient matching. *arXiv preprint arXiv:2006.05929*,

2020.
Zheng, X.-Y., Lee, M.-C., and Hong, Y.-W. P. Knowledge caching for federated learning. In *2021 IEEE Global* *Communications Conference (GLOBECOM)*, pp. 1–6,

2021. doi: 10.1109/GLOBECOM46510.2021.9685861.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al. Lima: Less is more for alignment. *Advances in Neural Information Processing* *Systems*, 36, 2024.

<!-- page 12 -->

## A. Details about the BDL

### A.1. Proof for BDL

First, we discuss the monotonicity with respect to *qθ/p* for the following equation:

<u>Pm(1 − λ)p + λqθ</u> =*,* (8) *Qmλp* + (1 *− λ*)*qθ*

<u>qθ</u> **Proof.** To simplify the analysis, let *x* = (representing the relative ratio of *qθ*and *p*). Then we have *p*

<u>Pm(1 − λ) + λx</u> =*.* (9) *Qmλ* + (1 *− λ*)*x*

<u>Pm</u> Next, differentiate with respect to *x*: *Qm*

<u>Pm</u> <u>∂</u> <u>Qmλ(λ + (1 − λ)x) − ((1 − λ) + λx)(1 − λ)</u> = (10) *∂x* (*λ* + (1 *− λ*)*x*)2

<u>2λ − 1</u> =*,* (11) (*λ* + (1 *− λ*)*x*)2

2 where the denominator (*λ* + (1 *− λ*)*x*) is always positive, and the numerator is a constant 2*λ −* 1. When *λ >* 0*.*5, *Pm* <u>∂Qm Pm</u> 2*λ −* 1 *>* 0, which implies *>* 0, meaning that is monotonically increasing with respect to *x*. On the contrary, *∂x Qm* <u>Pm Pm</u> when *λ <* 0*.*5, is monotonically decreasing. And when *λ* = 0*.*5, is a constant. Due to the monotonic function of *Qm Qm* *Pm Pm λλ* Eq. (9), it is easy to find out the boundary values of

|  |  |  |  | . As x → 0 (i.e., q | ≫ p), |  | →; while as x →∞ (i.e., |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P 1− |  |  | Q |  | θ | Q | 1− |
| Q λλ |  | 1 − 𝜆 𝜆 log 𝜆 | 1 +2− 𝜆 |  | 𝜆 𝜆 log 1 − 𝜆 |  |  |

*Qmθ Qm* 1*−* <u>Pm 1−</u> *qθ≪ p*), *→*. *m*

*Figure 5.* Curves of the two boundaries of *C*(*x*).

Next, we prove the monotonicity of *C*(*x*) in Eq. (5) that

<u>Pm(x) Pm(x)</u> *C*(*x*) = *λ*log + *λ −* (1 *− λ*) (12) *Qm*(*x*) *Qm*(*x*)

**Proof.** We first differentiate *C*(*x*) as <u>∂C(x) λ</u> = *−* (1 *− λ*)*,* (13) *∂z z* where *z* = *Pm/Qm*. And the second derivative of *C*(*x*) is

<u>∂ C(x) λ</u> = *−,* (14) *∂z z*

![PDF image 1 on page 12](asset:sha256:0ae12e7640aa6e0ae54dc05543187fde93888b7a244ce55e8a92b61964327035)

<!-- page 13 -->

which constantly below 0 since *λ >* 0. Therefore, when *z* = 1*−* *λλ*, i.e., *Q* *Pm* *m* = 1*−* *λλ*, *C*(*x*) reaches its maximum *λ*log 1*−* *λλ*. When *λ <* 0*.*5, *λ*log

| < 0, constantly brings negative C(x). It always makes Q |  |  |  | to increase and match P |  | , which is |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  | 1− λλ | 1 λ | λλ 1− |  |

1*−* <u>λ</u> *λ m m* how standard KL Divergence works. However, when *λ >* 0*.*5, *C*(*x*) *∈* (*λ*log 1*−* + 2 *−* 1 *,λ*log *λλ* ). By drawing the curves of the two boundaries in Figure5, we can see that when 0*.*5 *< λ <∼* 0*.*7 (value that around 0*.*7), *C*(*x*) is constantly positive, which is how reversed KL Divergence works in most cases. However, if *∼* 0*.*7 *< λ <* 1, the range of *C*(*x*) includes both positive and negative values. Since we blend probabilities *qθ*and *p* into *Pm*and *Qm*, it can be regarded as a combination of KL and reversed KL. And it is easy to imagine that when a larger *λ* is applied, the range of *C*(*x*) is larger, but if *λ* is close to 1, it behaves similar to reversed KL Divergence.

In a word, we can draw two conclusions. First, given a certain constant *λ*, *C*(*x*) falls in the range between *λ*log <u>1−</u> *λ* <u>λ</u> + 2*−* *λ* <u>1</u>

and *λ*log 1*−* <u>λ</u> *λ*, which provides finite and known coefficients on the gradients *∇θqθ*(*x*). It avoids the gradient explosion issue during training in case of the outliers in student models’ output probability. Second, when the predefined *λ ∈* (0*.*7*,*1), *C*(*x*) can be positive or negative values, allowing the *qθ*to increase or decrease flexibly. While when being close to 1, it has the probability to give an extreme regularization.

**A.2. Further Discussion on Parameter** *λ* Based on above conclusion, we can understand how the loss function BDL functions within the method: when the difference of teacher and student model enlarges regarding to sample *x*, the output probabilities of the two networks differs, leading to a larger *qθ/p*. And through the blending of probabilities in BDL, the gradients in the backpropagation will be enhanced by a larger coefficient *C*(*x*), which emphasizes the importance of the sample *x*. Based on our analysis above, moderate *λ* is required. Our parameter analysis experiments in Sec.4.3.2show that BDL consistently brings improvements of most *λ* except 0*.*5. It is easy to understand that *C*(*x*) has the smallest range near 0 when *λ* = 0*.*5, which leads to the gradient vanishing issue.

| P |  |  |
| --- | --- | --- |
| Q | qp |  |
| P | qp |  |
| Q |  |  |
| P |  | P |
| Q |  | Q |

 When *λ ∈* (0*,*0*.*5),
<u>m</u> *m* is closer to *θ*, and *C*(*x*) is negative, making BDL behaves similar to standard KL divergence. When *λ ∈* (0*.*5*, ∼* 0*.*7), <u>m</u> *m* transits to <u>θ</u>, and *C*(*x*) takes positive values to make *qθ*increase, making BDL behaves similar to reverse KL divergence. However, *λ* = 0*.*9 excels the best performance, which demonstrates that the combination of KL and reverse KL helps the training and convergence of the model. Meanwhile, a larger *λ* allows a larger range of *C*(*x*), and also brings a bigger <u>m</u> *m*. Based on the monotonicity of <u>m</u> *m* and *C*(*x*), we can draw that samples has distinct output probabilities from teacher and student will have a larger *C*(*x*) with a larger *λ*, which means the gradient will be enlarged and emphasized.

Based on this, we hypothesize that our method inherently addresses some intrinsic issues in existing distillation frameworks, such as stabilizing gradients and emphasizing hard samples. However, determining the optimal value of *λ* remains an open question.