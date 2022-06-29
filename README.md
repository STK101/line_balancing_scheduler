# line_balancing_scheduler
Use 
```python starter.py --unsequenced_schedule "Path to the Input Excel File" --file_name "Name of the output excel file with xlsx extentions"```

eg - ```python starter.py --unsequenced_schedule "UNSEQUENCE PLAN 26 & 27 may priority.xlsx" --priority_present "True" --file_name "output_p.xlsx" ```

To look for other run options look into the starter.py -> parse_args() function 

# Tracker
1) Fix Clutter - Done
2) Better Priority Handling - Done
3) Fix output from Tool - Ongoing
4) APC handling - Ongoing
5) Line 1 and Line 2 Division of SKUs - Not Started

## APC Handling
a. Just before some packet - Super Sku
b. X hours before some packet - Need to add a time component to our paint scheduler (need time required to paint an sku + changeover time etc to predict the right time to make APC)
