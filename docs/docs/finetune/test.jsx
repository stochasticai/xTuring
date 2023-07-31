import React, { useState } from 'react'
import clsx from 'clsx'
import styles from './styles.module.css'
import MDXContent from '@theme/MDXContent'
import CodeBlock from '@theme/CodeBlock'
    
export default function Test() {
  const [model, setModel] = useState('LLaMa')
  const [version, setVersion] = useState('LLaMa')

  console.log(model, version)

  return (
    <div className={clsx('col')}>
      <label htmlFor='model'>Choose a model:</label>

      <select
        name='model'
        id='model'
        onChange={(e) => console.log(e.target.value)}
      >
        <option value='LLaMa'>LLaMa</option>
        <option value='GPT-J'>GPT-J</option>
        <option value='GPT-3'>GPT-3</option>
        <option value='GPT-4'>GPT-4</option>
      </select>

      <label htmlFor='model-version'>Choose a version:</label>

      <select
        name='model-version'
        id='model-version'
        onChange={(e) => console.log(e.target.value)}
      >
        <option value='v1'>V1</option>
        <option value='v2'>V2</option>
        <option value='v3'>V3</option>
        <option value='v4'>V4</option>
      </select>

      <CodeBlock
        className='row'
        showLineNumbers={false}
        language='python'
        children={`import json from datasets 
import Dataset, DatasetDict

alpaca_data = json.load(open(alpaca_dataset_path)) 

instructions = []
inputs = [] 
outputs = [] 

for data in alpaca_data:
    instructions.append(data["instruction"]) inputs.append(data["input"])
    outputs.append(data["output"])

data_dict = {
    "train": {"instruction": instructions, "text": inputs, "target": outputs}
} 

dataset = DatasetDict() 

for k, v in data_dict.items(): 
    dataset[k] =
    Dataset.from_dict(v) dataset.save_to_disk(str("./alpaca_data"))
`}
      />
    </div>
  )
}