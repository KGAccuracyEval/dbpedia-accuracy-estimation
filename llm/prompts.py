CORRECTNESS_PROMPT = """
Instructions:
1. You will be presented with a STATEMENT in a subject predicate object (SPO) triplet format. 
2. Your task is to answer whether the presented STATEMENT is correct or incorrect based on your training data. 
3. There is also the option to mark I don’t know (IDK) in case you are not sure.
4. You need to assume that we’re in the year 2015. It is probable that a given fact was correct in 2015, but not anymore. For this specific task, it should be marked as correct.
5. Some examples have been provided for you to learn how to do this task. Your task is to provide a RESPONSE for the STATEMENT under "Your Task".  
6. Your RESPONSE must be either "Correct", "Incorrect", or "IDK".

Example1:
STATEMENT:
Barack_Obama currentPresident USA

RESPONSE:
Correct

Example2:
STATEMENT:
Cristiano_Ronaldo playsForTeam Juventus

RESPONSE:
Incorrect

Example3:
STATEMENT:
Toni_Sugaman ownerOf Towels_&_Sinks_s.r.l.

RESPONSE:
IDK

Your Task:
STATEMENT:
{fact}

RESPONSE:
"""

RETRY_PROMPT = """
You did not respond with one of the three allowed labels: "Correct", "Incorrect", or "IDK". 
You still have {chances} chances left to answer with one of the three allowed labels.

Your Task:
STATEMENT:
{fact}

RESPONSE:
"""

SYSTEM_PROMPT = """
You are a helpful assistant that helps identify if a given STATEMENT is correct or not.
If you don't have evidence about the STATEMENT in your training data, please don't share false information.
"""
