import multiprocessing

# Example function to execute
def process_item(item):
    return item * item

# List of items to process
items = [1, 2, 3, 4, 5]

# Using multiprocessing Pool to run the loop in parallel
if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_item, items)

    print(results)
