This repository contains the code for the Editor, which is a local example used in our submitted paper. Specially, Editor is a novel multi-resolution framework for MTS data cleaning that include detection, localization and repair modules:<br>
>Detection Module. Editor identifies erroneous windows containing errors across varying magnitudes.<br>
Localization Module. Editor pinpoints multi-granularity errors within the erroneous window.<br>
Repair Module. Editor generates contextually consistent corrections for the localized errors.<br>

<h3><strong>🛠️ Prerequisites</strong></h3><br>
This is the configuration required for the Editor runtime environment.<br>
To install the required packages, you can create a conda environment:<br>

<pre style="background-color: #f0f0f0;">
<code>
conda create --name Editor python=3.8
</code>
</pre>
then use pip to install the following libraries:<br>
<pre style="background-color: #f0f0f0;">
<code>
torch 1.8.0<br>
numpy 1.24.4<br>
pandas 2.0.3<br>
scikit-learn 1.3.2<br>
scipy 1.10.1<br>
</code>
</pre>

<h3><strong>:package: Datasets</strong></h3><br>
SWaT、WADI、PUMP、SMD、MSL are in the folder "input".<br>

<h3><strong>:rocket: Start</strong></h3><br>
If you want to execute the detection module, please run detection_step1, detection_step2, and detection_step3 in order, please use the following command:<br>
<pre style="background-color: #f0f0f0;">
<code>
python detection_step#.py
</code>
</pre>

Then, if you want to execute the localization module, please run localization_step, please use the following command:<br>
<pre style="background-color: #f0f0f0;">
<code>
python localization_step.py
</code>
</pre>

Finally, if you want to execute the repair module, please run repair_step, please use the following command:<br>
<pre style="background-color: #f0f0f0;">
<code>
python repair_step.py
</code>
</pre>

EDITOR can also be configured to place all modules in a single Python file for cleaning execution as needed.

Additionally, Parameters and architectures can be modified according to your own situation.
