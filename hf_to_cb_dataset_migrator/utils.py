def generate_help(description, examples):
    help_text = f"{description}\n\n\b\nExamples:\n\n"
    for i, line in enumerate(examples):
        if i!=0:
           help_text+="\n\n"  
        help_text += "  "+line
    #print(help_text)    
    return help_text