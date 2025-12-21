# fmt:off

# ===============================================
# LINK: Evaluation Criterias
# ===============================================
eval_criterias = {
#--------------------------------
# Likert evaluation criteria
#--------------------------------

"bin": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"bin_unc": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
UNCERTAIN: You are uncertain about the degree of accomplishment of the objective. The task can be fully accomplished, or not accomplished.
""",


"tri": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"four_room": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
SUCCESS WITH ROOM FOR IMPROVEMENT: The assistant executed **all of** what's necessary to complete the objective, although it may have been inefficient and made some mistakes. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"four_room_2": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
SUCCESS WITH ROOM FOR IMPROVEMENT: The assistant executed **all of** what's necessary to complete the objective, although it may have taken an inefficient route and made some mistakes. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",


"four_excellent": """
EXCELLENT: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
GOOD: The assistant executed **all of** what's necessary to complete the objective, although it may have been inefficient and made some mistakes. The task is fully accomplished.
FAIR: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
POOR: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"four_excellent_2": """
EXCELLENT: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
GOOD: The assistant executed **all of** what's necessary to complete the objective, although it may have been inefficient and made some mistakes. The task is fully accomplished.
FAIR: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
POOR: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"four_letter": """
A: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
B: The assistant executed **all of** what's necessary to complete the objective, although it may have been inefficient and made some mistakes. The task is fully accomplished.
C: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
D: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"four_letter_2": """
A: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
B: The assistant executed **all of** what's necessary to complete the objective, although it may have been inefficient and made some mistakes. The task is fully accomplished.
C: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
D: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",


"tri_unc": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
UNCERTAIN: You are uncertain about the degree of accomplishment of the objective. The task can be fully accomplished, partially accomplished, or not accomplished.
""",

"four_unclear": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
PARTIAL FAILURE: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
UNCLEAR: You are uncertain if the assistant's work is correct or incorrect.""",

"four": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
PARTIAL FAILURE: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"four_unc": """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
PARTIAL FAILURE: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
UNCERTAIN: You are uncertain about the degree of accomplishment of the objective. The task can be fully accomplished, partially accomplished, or not accomplished.
""",

#------------------------------------
# LINK Numeric evaluation criteria
#------------------------------------

"bin_num_unc": """Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.

Score 1 to 4: You are confident that the assistant's work is incorrect, but there is a small chance it could be partially correct.
Score 5 to 6: You are uncertain about the correctness of the assistant's work; it could be right or wrong.
Score 7 to 10: You are confident that the assistant's work is completely correct for the task.
""",

"tri_num_accomplished_inv": """Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.

Score 1 to 4: You are confident that the assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
Score 5 to 6: You are confident that the assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
Score 7 to 10: You are confident that the assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",


"tri_num_accomplished": """Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.

Score 1 to 4: You are confident that the assistant executed  **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
Score 5 to 6: You are confident that the assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
Score 7 to 10: You are confident that the assistant executed **most of** what's necessary to complete the objective. The task is fully accomplished.
""",

"four_num_unc": """Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.

Score 1 to 2: You are very confident that the assistant's work is completely incorrect for the task.
Score 3 to 4: You are fairly confident that the assistant's work is incorrect, but there is a small chance it could be partially correct.
Score 5 to 6: You are uncertain about the correctness of the assistant's work; it could be right or wrong.
Score 7 to 8: You are fairly confident that the assistant's work is correct, but there is a small chance it could be partially incorrect.
Score 9 to 10: You are very confident that the assistant's work is completely correct for the task.
""",

"four_num_unc_accomplished_inv": """Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.

Score 1 to 2: You are very confident that the assistant's work is completely correct for the task.
Score 3 to 4: You are fairly confident that the assistant's work is correct, but there is a small chance it could be partially incorrect.
Score 5 to 6: You are uncertain about the correctness of the assistant's work; it could be right or wrong.
Score 7 to 8: You are fairly confident that the assistant's work is incorrect, but there is a small chance it could be partially correct.
Score 9 to 10: You are very confident that the assistant's work is completely incorrect for the task.
""",

"four_num_unc_accomplished": """Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.
Score 1 to 2: You are very confident that the assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
Score 3 to 4: You are fairly confident that the assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
Score 5 to 6: You are uncertain if the assistant executed what's necessary to complete the objective. The task can be accomplished or not accomplished.
Score 7 to 8: You are fairly confident that the assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
Score 9 to 10: You are very confident that the assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
""",

"range_num": """
Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.
The score can be any scalar between 1 and 10, where:
- A score close to 1 means you are very confident that the assistant's work is completely incorrect for the task.
- A score close to 10 means you are very confident that the assistant's work is completely correct for the task.
- A score close to 5 means you are uncertain about the correctness of the assistant's work; it could be right or wrong.
""",

"range_num_vague": """
Provide a single score, rating the assistant's work strictly on a scale of 1 to 10 based on the confidence of its correctness for the task.
The score can be any scalar between 1 and 10, where:
- A **low score** means you are very confident that the assistant's work is completely incorrect for the task.
- A **high score** means you are very confident that the assistant's work is completely correct for the task.
- A **mid-range score** means you are uncertain about the correctness of the assistant's work; it could be right or wrong.
""",

"range_num_confidence_0_1": """
Provide a number between 0 and 1 reflecting your confidence that the assistant's work is correct for the task.
""",

"range_num_likelihood_0_100": """
Provide a single number between 0 and 100 representing the likelihood that the assistant's work is correct for the task.
""",

"range_num_likelihood_0_100_2": """
Provide a single number between 0 and 100 indicating the likelihood that the trajectory fully and successfully completes the objective.
""",

"four_degree": """
Provide a single score strictly on a scale of 1 to 10 reflecting the degree of accomplishment of the objective.

Score 1 to 2: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
Score 3 to 4: The assistant executed **a few of** what's necessary to complete the objective. The task is partially accomplished.
Score 5 to 6: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
Score 7 to 8: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
Score 9 to 10: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
""",

#TODO
"four_degree_vague": """
Provide a single score strictly on a scale of 1 to 10 reflecting the degree of accomplishment of the objective.

The score can be any scalar between 1 and 10, where:
- A **low score** means the task is mostly not accomplished.
- A **mid-range score** means the task is partially accomplished.
- A **high score** means the task is mostly accomplished.
""",

"four_room_score": """
100: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
75: The assistant executed **all of** what's necessary to complete the objective, although it may have taken an inefficient route and made some mistakes. The task is fully accomplished.
50: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
0: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

"four_room_score_2": """
100: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
75: The assistant executed **all of** what's necessary to complete the objective, although it may have taken an inefficient route and made some mistakes. The task is fully accomplished.
50: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
0: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
""",

#--------------------------------
# LINK Other criteria
#--------------------------------

"three_statement_unc_letter": """
Here are statements about the assistant's work. Please reply with the option that best describes the assistant's work.
Which statement best describes the assistant's work?
A. The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
B. The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
C. The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
D. It is not clear if the assistant executed what's necessary to complete the objective. The task can be fully accomplished, partially accomplished, or not accomplished.
""",

"four_statement_unc_letter": """
Here are statements about the assistant's work. Please reply with the option that best describes the assistant's work.
Which statement best describes the assistant's work?
A. The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
B. The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
C. The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
D. The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
E. It is not clear if the assistant executed what's necessary to complete the objective. The task can be fully accomplished, partially accomplished, or not accomplished.
""",


"four_statement_unc_letter_t_f": """
The following are statements about the assistant's work. If a statement is correct, please reply with T, otherwise reply with F.

A. The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
B. The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished. 
C. The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
D. The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
E. It is not clear if the assistant executed what's necessary to complete the objective. The task can be fully accomplished, partially accomplished, or not accomplished.

Provide your response as a list of letters for each statement, as follows:
T or F, T or F, T or F, T or F, T or F
""",

"three_option_unc": """Here are three statements and only one of them corresponds to the options. Please return the option with the correct answer. If there is no correct answer in the options, please reply with N.
I. The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
II. The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
III. The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.""",

"four_option_unc_question": """Please evaluate the assistant's work by selecting the option that best represents your level of confidence.
What best describes your confidence in the assistant's work?
I. You are very confident that the assistant's work is completely incorrect for the task.
II. You are fairly confident that the assistant's work is incorrect, but there is a small chance it could be partially correct.
III. You are uncertain about the correctness of the assistant's work; it could be right or wrong.
IV. You are fairly confident that the assistant's work is correct, but there is a small chance it could be partially incorrect.
V. You are very confident that the assistant's work is completely correct for the task.
""",

}


eval_prompt_templates = {
"ins": 
"""Now please provide your response.


## Here are the evaluation instructions:
{eval_criteria}

## Provide your response as follows:
{response_format}""",


"cat": 
"""Now please provide your response.


## Here is the evaluation criteria:
{eval_criteria}

## Provide your response as follows:
{response_format}
""",
}


noise_injections = {
    "latin_noise": """Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Ut purus elit, vestibulum ut, placerat ac, adipiscing vitae, felis. Curabitur dictum gravida mauris. Nam arcu libero, nonummy eget, consectetuer id, vulputate a, magna. Donec vehicula augue eu neque. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Mauris ut leo. Cras viverra metus rhoncus sem. Nulla et lectus vestibulum urna fringilla ultrices. Phasellus eu tellus sit amet tortor gravida placerat."""
}



response_format = f"""
{{k_step}}
{{cot_parts}}
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the assistant can progress towards the objective OR "NA" if not applicable]"""


# response_format = f"""
# {{k_step}}
# {{cot_parts}}
# VERDICT: [Your verdict following the evaluation criteria]
# FEEDBACK: [Feedback so the assistant can progress towards the objective OR "NA" if not applicable]"""

eval_prompt_template = f"""Now please provide your response.
\n
## Here is the evaluation criteria:
{{eval_criteria}}
\n
## Provide your response as follows:
{{response_format}}
"""

common_prompts = {
"cot_phrase_short": """REASONING: [Comprehensive step-by-step reasoning to come up with your evaluation and feedback]""",


"cot_phrase": """REASONING: [Comprehensive step-by-step reasoning over all evidence and information to come up with your evaluation and feedback]""",

"cot_phrase_thoughts": """THOUGHTS: [Your thoughts and reasoning process to come up with your evaluation and feedback]""",

"eval_criterias": eval_criterias,

"eval_prompt_templates": eval_prompt_templates,

"noise_injections": noise_injections,

"response_format": response_format,

"eval_prompt_template": eval_prompt_template,

}
