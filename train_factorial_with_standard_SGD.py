# THIS SCRIPT TRAINS A PYTHIA MODEL WITH COT FACTORIAL DATA WITH STANDARD SGD
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
verbose = True
total_iterations = 10000
num_iters_per_validation = 100
# max_gen = 4735 # this is the size of format_sample(130) i.e. 130!... but it's going to explode VRAM! :( 
# max_gen = 40 # try this one?
train_numbers = list(range(1, 10)) + list(range(15, 20))
test_numbers = list(range(10, 15)) + list(range(20, 25))
############################################################################################################################################
############################################################################################################################################
#######
# Import necessary libraries
import torch
import pdb
import torch.nn as nn
from transformers import GPTNeoXForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers import StoppingCriteria, StoppingCriteriaList

import numpy as np
import math
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time

def preprocess_for_generation(input_ids, attention_mask):
    # Find the position of the <sep> token
    sep_token_id = tokenizer.convert_tokens_to_ids('<sep>')
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    
    if len(sep_positions) > 0:
        # Cut off everything after the first <sep> token (inclusive)
        cut_position = sep_positions[0] + 1  # Include the <sep> token
        input_ids = input_ids[:cut_position]
        attention_mask = attention_mask[:cut_position]
    else:
        # If <sep> not found, use the whole input_ids
        print("heree!!!")
        pass  # Handle accordingly if necessary
    
    return input_ids, attention_mask

def format_sample_no_COT(n):
    # factorial_value = compute_factorial(n)
    # # Get 7 significant figures
    # factorial_str = f"{factorial_value:.6e}"
    # # Ensure it has 7 significant figures
    # significant_figures = f"{float(factorial_str):.7g}"
    significant_figures = format_decimal(factorial_value)
    sample = f"<bos> {n}! <sep> #### {significant_figures} <eos>"
    return sample
    
def format_sample_with_instructions_and_COT(n):
    from decimal import Decimal, getcontext
    getcontext().prec = 1000  # Set precision high enough to handle large factorials
    steps = []
    current_total = Decimal(n)
    for i in range(n - 1, 0, -1):
        i_decimal = Decimal(i)
        next_total = current_total * i_decimal
        # Format the numbers with 7 significant figures
        current_total_str = format_decimal(current_total)
        next_total_str = format_decimal(next_total)
        step_str = f"{current_total_str}*{i}=<multiply>{next_total_str}"
        steps.append(step_str)
        current_total = next_total
    steps_str = " <sep> ".join(steps)
    # Format final answer with 7 significant figures
    final_answer = format_decimal(current_total)
    sample = f"<bos> Calculate factorial which is defined as: n!=n*(n-1)*(n-2)...(3)(2)(1). You can break it up in steps so only multiply 2 numbers at a time. Solve: {n}! <sep> {steps_str} <sep> so the answer is {final_answer}. #### {final_answer} <eos>"
    return sample

def format_sample_with_COT(n):
    from decimal import Decimal, getcontext
    getcontext().prec = 1000  # Set precision high enough to handle large factorials
    steps = []
    current_total = Decimal(n)
    for i in range(n - 1, 0, -1):
        i_decimal = Decimal(i)
        next_total = current_total * i_decimal
        # Format the numbers with 7 significant figures
        current_total_str = format_decimal(current_total)
        next_total_str = format_decimal(next_total)
        # Include the <multiply> token and the result
        step_str = f"{current_total_str}*{i}=<multiply>{next_total_str}"
        steps.append(step_str)
        current_total = next_total
    steps_str = " <sep> ".join(steps)
    # Format final answer with 7 significant figures
    final_answer = format_decimal(current_total)
    sample = f"<bos> {n}! <sep> {steps_str} <sep> so the answer is {final_answer}. #### {final_answer} <eos>"
    return sample


def format_decimal(num):
    return  f"{float(num):.7g}"
    # num_str = format(num, '.7g')
    # return num_str


format_sample = format_sample_with_COT # format_sample_with_instructions_and_COT / format_sample_no_COT / format_sample_with_COT

# Set device to GPU cuda:0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seed for reproducibility
random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-410m')
model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-410m')

# Add special tokens to the tokenizer, including '<multiply>'
tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'bos_token': '<bos>',
    'sep_token': '<sep>',
    'eos_token': '<eos>',
    'additional_special_tokens': ['<multiply>']
})

# Update the model's embeddings to accommodate the new tokens
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id 
model.config.eos_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id 

# Wrap the model with DataParallel to use multiple GPUs
device_ids = [0, 1]  # GPUs cuda:0 and cuda:1
model = torch.nn.DataParallel(model, device_ids=device_ids)

model.to(device)
# model.to('cpu')  # Move to CPU first to avoid memory issues
    
# Prepare the dataset
def compute_factorial(n):
    return math.factorial(n)

# Generate training data

train_samples = [format_sample(n) for n in train_numbers]

# Generate test data

test_samples = [format_sample(n) for n in test_numbers]

def calculate_per_token_accuracy(test_logits, test_labels):
    
        shift_logits = test_logits[..., :-1, :].contiguous()
        shift_predictions = shift_logits.argmax(dim=-1)
        
        shift_labels = test_labels[..., 1:].contiguous()
        
        # Create a mask to ignore padding tokens in labels
        shift_mask = shift_labels != tokenizer.pad_token_id
        
        # Compute correct predictions
        shift_correct = (shift_predictions == shift_labels) & shift_mask
        
        # Calculate per-token accuracy
        per_token_accuracy = shift_correct.sum().item() / shift_mask.sum().item()
        return per_token_accuracy

# Tokenize the data
def tokenize_samples(samples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for sample in samples:
        # Tokenize the sample with offsets mapping
        encoded = tokenizer.encode_plus(
            sample,
            return_tensors='pt',
            padding=False,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]
        offsets = encoded['offset_mapping'][0]
        
        # Initialize labels as the same as input_ids
        labels = input_ids.clone()
        
        # Get the text
        text = sample
        
        # Find the positions of '<multiply>' in the tokenized input_ids
        multiply_token_id = tokenizer.convert_tokens_to_ids('<multiply>')
        multiply_positions = (input_ids == multiply_token_id).nonzero(as_tuple=True)[0]
        
        for pos in multiply_positions:
            # The character position corresponding to this token
            start_char = offsets[pos][0].item()
            end_char = offsets[pos][1].item()
            
            # From end_char onwards, find the position of the multiplication result
            # We'll assume the multiplication result is the number immediately after '<multiply>'
            # So we can use regex to find the number starting from end_char
            match = re.search(r'\s*([\d\.e\+\-]+)', text[end_char:])
            if match:
                number_start = end_char + match.start()
                number_end = end_char + match.end()
                
                # Now, find the tokens that correspond to this range
                token_positions = []
                for idx, (token_start, token_end) in enumerate(offsets):
                    token_start = token_start.item()
                    token_end = token_end.item()
                    if token_start >= number_end:
                        break
                    if token_end > number_start and token_start < number_end:
                        token_positions.append(idx)
                # Set labels at these positions to -100
                labels[token_positions] = -100
            else:
                # No number found after '<multiply>'
                pass
        
        # Append to the lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    # Pad the sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    return {'input_ids': input_ids_padded, 'attention_mask': attention_mask_padded, 'labels': labels_padded}

train_data = tokenize_samples(train_samples)
test_data = tokenize_samples(test_samples)

# Move data to device
for key in train_data:
    train_data[key] = train_data[key].to(device)
for key in test_data:
    test_data[key] = test_data[key].to(device)

# Set up optimizer, loss, and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(.9,.99),  weight_decay=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=4, num_training_steps=total_iterations)

# Initialize metrics storage
train_losses = []
test_losses = []
train_per_token_accuracies = []
test_per_token_accuracies = []
train_final_answer_accuracies = []
test_final_answer_accuracies = []
train_iterations = []
test_iterations = []

# Optimized generate_with_calculator function
def generate_with_calculator(model, tokenizer, input_ids, attention_mask, max_length, device):
    with torch.no_grad():
        generated_ids = input_ids.clone().to(device)
        multiply_token_id = tokenizer.convert_tokens_to_ids('<multiply>')
        eos_token_id = tokenizer.eos_token_id
        sep_token_id = tokenizer.convert_tokens_to_ids('<sep>')

        # Process the initial input_ids to get past_key_values
        outputs = model(
            input_ids=input_ids.to(device),
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        last_operator_pos = generated_ids.shape[1]

        for _ in range(max_length):
            # Generate next token
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            # Append the next token
            generated_ids = torch.cat((generated_ids, next_token_id.unsqueeze(0)), dim=1)

            # If the next token is '<multiply>', we need to compute the multiplication
            if next_token_id.item() == multiply_token_id:
                # Decode tokens from last_operator_pos to current position
                tokens_to_decode = generated_ids[0, last_operator_pos:]
                text_to_parse = tokenizer.decode(tokens_to_decode, skip_special_tokens=False)
                # Use regex to extract the operands immediately before '<multiply>'
                match = re.search(r'([\d\.e\+\-]+)\s*\*\s*([\d\.e\+\-]+)\s*=\s*<multiply>$', text_to_parse)
                if match:
                    operand1 = float(match.group(1))
                    operand2 = float(match.group(2))
                    result = operand1 * operand2
                    # Format the result
                    result_str = format_decimal(result) #f"{float(result):.7g}"
                    
                    # Tokenize the result
                    result_tokens = tokenizer.encode(result_str, return_tensors='pt').to(device)
                    # Append the result tokens to generated_ids
                    generated_ids = torch.cat((generated_ids, result_tokens[0].unsqueeze(0)), dim=1)
                    # Update past_key_values by passing through the result tokens
                    outputs = model(
                        input_ids=result_tokens.to(device),
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    # Update last_operator_pos
                    last_operator_pos = generated_ids.shape[1]
                    # Update next_token_logits for the next iteration
                    next_token_logits = outputs.logits[:, -1, :]
                    continue  # Proceed to the next iteration
                else:
                    # Cannot parse operands
                    pass
            elif next_token_id.item() == sep_token_id:
                # Update last_operator_pos
                last_operator_pos = generated_ids.shape[1]

            # Check for end of sequence
            if next_token_id.item() == eos_token_id:
                break

            # Get next token logits
            outputs = model(
                input_ids=next_token_id.unsqueeze(0).to(device),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
    return generated_ids


# # Define a custom generation function that uses the calculator multiply token
# def generate_with_calculator(model, tokenizer, input_ids, attention_mask, max_length, device):
#     generated_ids = input_ids.clone()
#     multiply_token_id = tokenizer.convert_tokens_to_ids('<multiply>')
    
#     for _ in range(max_length):
#         outputs = model(
#             input_ids=generated_ids.to(device),
#             attention_mask=attention_mask.to(device)
#         )
#         logits = outputs.logits
#         next_token_logits = logits[:, -1, :]
        
#         # Apply decoding strategy, e.g., sampling or argmax
#         next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
#         # Append the next token
#         generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
#         attention_mask = torch.cat((attention_mask, torch.ones_like(next_token_id)), dim=1)
        
#         # If the next token is '<multiply>', we need to compute the multiplication
#         if next_token_id.item() == multiply_token_id:
#             # Get the previous tokens to find the operands
#             text_so_far = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
#             # Use regex to extract the operands
#             match = re.findall(r'([\d\.e\+\-]+)\s*\*\s*([\d\.e\+\-]+)\s*=\s*<multiply>', text_so_far)
#             if len(match)>0:
                
#                 operand1 = float(match[-1][0])
#                 operand2 = float(match[-1][1])
                
#                 result = operand1 * operand2
#                 # Format the result
#                 result_str = f"{result:.7g}"
#                 # Tokenize the result
#                 result_tokens = tokenizer.encode(result_str, return_tensors='pt').to(device)
#                 # Append the result tokens to generated_ids
#                 generated_ids = torch.cat((generated_ids, result_tokens), dim=1)
#                 attention_mask = torch.cat((attention_mask, torch.ones_like(result_tokens[0])), dim=1)
#             else:
#                 # Cannot parse operands
#                 pass
        
#         # Check for end of sequence
#         if next_token_id.item() == tokenizer.eos_token_id:
#             break
    
#     return generated_ids

# Training loop
for iteration in tqdm(range(0, total_iterations + 1)):
    if iteration > 0:
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()
        
        outputs = model(**train_data)
        loss = outputs.loss.max()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Store train loss
        train_losses.append(loss.item())
        
        # Compute per token train accuracy
        with torch.no_grad():
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)

            # Compute per token train accuracy
            per_token_accuracy = calculate_per_token_accuracy(logits, train_data['labels'])
            train_per_token_accuracies.append(per_token_accuracy)
            train_iterations.append(iteration)
        
    # Every num_iters_per_validation iterations, evaluate and plot
    if iteration % num_iters_per_validation == 0:
        test_iterations.append(iteration)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(**test_data)
            test_loss = test_outputs.loss.max().item()
            test_losses.append(test_loss)
            
            # Compute per token test accuracy
            test_per_token_accuracy = calculate_per_token_accuracy(test_outputs.logits, test_data['labels'])
            test_per_token_accuracies.append(test_per_token_accuracy)
        
        print(f"generating on train: iter {iteration}")
        # Compute final answer accuracies
        # For train data
        train_final_correct = 0
        listt = list(range(len(train_samples)))
        for i in listt:#[:3]:
            # Original input_ids and attention_mask
            input_ids = train_data['input_ids'][i]
            attention_mask = train_data['attention_mask'][i]
            
            # Preprocess to exclude the answer
            input_ids, attention_mask = preprocess_for_generation(input_ids, attention_mask)
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            generated = generate_with_calculator(
                model.module,
                tokenizer,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=train_data['input_ids'][i].shape[0],
                device=device
            )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
            if verbose and iteration>1000:
                # Extract the answer
                print("="*50)
                print("TRAIN:")
                print(f"train truth: {train_samples[i]}")
                print(f"generated_text: {generated_text}")
                time.sleep(0.1)
            match = re.search(r'####\s*([\d\.e\+\-]+)', generated_text)
            if match:
                generated_answer = match.group(1)
                target_answer = re.search(r'####\s*([\d\.e\+\-]+)', train_samples[i]).group(1)
                if generated_answer.strip() == target_answer.strip():
                    train_final_correct += 1
        train_final_accuracy = train_final_correct / len(train_samples)
        train_final_answer_accuracies.append(train_final_accuracy)
        print(f"testing: iter {iteration}")
        # For test data
        test_final_correct = 0
        listt = list(range(len(test_samples)))
        for i in listt:#[:3]:
            # Original input_ids and attention_mask
            input_ids = test_data['input_ids'][i]
            attention_mask = test_data['attention_mask'][i]
            
            # Preprocess to exclude the answer
            input_ids, attention_mask = preprocess_for_generation(input_ids, attention_mask)

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            
            generated = generate_with_calculator(
                model.module,
                tokenizer,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=1000,
                device=device
            )
            
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
            if verbose: # and iteration>1000:
                print("="*50)
                print("TEST:")
                print(f"test truth: {test_samples[i]}")
                print(f"generated_text: {generated_text}")
                time.sleep(0.1)
                
            # Extract the answer
            match = re.search(r'####\s*([\d\.e\+\-]+)', generated_text)
            
            if match:
                generated_answer = match.group(1)
                target_answer = re.search(r'####\s*([\d\.e\+\-]+)', test_samples[i]).group(1)
                if generated_answer.strip() == target_answer.strip():
                    test_final_correct += 1
        test_final_accuracy = test_final_correct / len(test_samples)
        test_final_answer_accuracies.append(test_final_accuracy)
        
        # Print metrics
        print(f"Iteration {iteration}")
        if iteration > 0:
            print(f"Train Loss: {loss.item():.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Per Token Train Accuracy: {per_token_accuracy*100:.2f}%")
            print(f"Per Token Test Accuracy: {test_per_token_accuracy*100:.2f}%")
            print(f"Final Answer Train Accuracy: {train_final_accuracy*100:.2f}%")
            print(f"Final Answer Test Accuracy: {test_final_accuracy*100:.2f}%")
        
        # Plot the metrics
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(train_iterations, train_losses, label='Train Loss')
        plt.plot(test_iterations, test_losses, label='Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss ')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(train_iterations, train_per_token_accuracies, label='Train Per Token Accuracy')
        plt.plot(test_iterations, test_per_token_accuracies, label='Test Per Token Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Per Token Acc')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(test_iterations, train_final_answer_accuracies, label='Train Final Answer Accuracy')
        plt.plot(test_iterations, test_final_answer_accuracies, label='Test Final Answer Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Final Answer Acc')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
