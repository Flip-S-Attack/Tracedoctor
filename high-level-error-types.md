
# LLM Log Parsing Error Types Analysis Report

## Overview

### 1. Variable Segmentation Errors - 22.6%
Models typically decompose compound single variables into multiple variables, such as treating IP addresses and port numbers as separate variables instead of viewing them as a single entity.

**Example**: `192.168.1.1:8080` is incorrectly parsed as two variables `<*>:<*>` instead of a single variable `<*>`

### 2. Contextual Over-analysis - 20.4%
Models over-interpret certain nouns or numbers, identifying them as variables or constants while ignoring their semantic context.

**Example**: In `connect to server 3 times`, incorrectly parsing `3` as a variable

### 3. Over-specification of Variable Names - 18.8%
During output, models parse numeric variables into specific pronouns, such as interpreting IP addresses as `<IP>` instead of the standard `<*>`.

**Example**: Output `Connection from <IP>` instead of `Connection from <*>`

### 4. Key-value Template Mishandling - 16.2%
When encountering key-value pair variables, models typically treat the entire pair as one variable, such as incorrectly identifying `port:400` as a single variable.

**Example**: In `config port:8080 timeout:30`, treating `port:8080` and `timeout:30` each as single variables

### 5. Non-standard Output Format - 6.7%
When outputting parsed logs, models may break the original log structure, such as introducing symbols that don't exist in the original logs.

**Example**: Adding extra brackets or separators in templates

### 6. Structural Bracket Preservation Failure - 6.4%
When parsing variables like `(<*>)`, models typically include brackets as part of the variable during replacement.

**Example**: `(timeout=30)` is parsed as `(<*>)` instead of `(<*>)`

### 7. Timestamp Component Handling Errors - 0.8%
Failure to correctly identify and handle individual components of complex timestamp formats.

### 8. Unit/Context Preservation Failures - 0.7%
Ignoring numerical units or contextual information, leading to inaccurate parsing.

### 9. Protocol/Service Identifier Confusion - 0.6%
Incorrectly identifying or classifying network protocols and service identifiers.

### 10. Block Variable Missed - 0.6%
Failing to recognize variable combinations that should be treated as a single block.

### 11. Pattern Over-Reliance - 0.5%
Over-dependence on learned patterns, unable to adapt to new log formats.

### 12. Incorrect Handling of Time Duration Variables - 0.5%
Incorrectly processing variables representing time intervals or durations.

### 13. Identifier Preservation Errors - 0.5%
Failure to correctly preserve important system identifiers.

### 14. Bracketed Content Misclassification - 0.4%
Incorrectly classifying the type of content within brackets.

### 15. False Variable Introduction - 0.4%
Introducing false variables where variables should not exist.

### 16. Pattern Over-segmentation Issues - 0.4%
Log patterns being excessively subdivided, losing their original structural integrity

### 17. Pattern Under-consolidation Problems - 0.3%
Insufficient pattern consolidation, unable to form effective unified structures

### 18. Value Retention in Templates - 0.3%
Incorrectly retaining specific values in templates that should be abstracted as variables.

### 19. Path/Identifier Handling Errors - 0.3%
Incorrectly processing file paths or system identifiers.

### 20. Address-Port Unitization Failure - 0.3%
Failure to treat addresses and ports as unified units.

### 21. Fixed-Term Hesitation - 0.2%
Unnecessary variabilization of terms that should remain fixed.

### 22. Source Information Fragmentation - 0.2%
Incorrectly fragmenting continuous source information.

### 23. Request Line Misaggregation - 0.2%
Incorrectly processing HTTP request lines or similar structures.

### 24. Domain Parsing Incompleteness - 0.2%
Inability to completely parse domain name structures.

### 25. Fixed Tag Omission - 0.2%
Omitting fixed tags that should be preserved.

### 26. Formatting Deviation - 0.2%
Output format deviating from expected standards.

### 27. Escape Sequence Misplacement - 0.1%
Incorrectly processing or placing escape sequences.

### 28. Source Data Grouping Errors - 0.1%
Incorrectly grouping related source data.

### 29. Under-templatization of Dynamic Elements - 0.1%
Insufficient processing of dynamic elements that should be templatized.