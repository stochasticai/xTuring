import React, { useState } from 'react'
import clsx from 'clsx'
import MDXContent from '@theme/MDXContent'
import CodeBlock from '@theme/CodeBlock'

export default function Test(
  {instruction}
) {
  const [code, setCode] = useState('llama');

  return (
    <div className={clsx('col')}>
      <label htmlFor='model'>Choose a model: </label>

      <select
        name='model'
        id='model'
        onChange={(e) => setCode(e.target.value)}
      >
        <option value='llama'>LLaMA</option>
        <option value='llama_lora'>LLaMA LoRa</option>
        <option value='llama_lora_int8'>LLaMA LoRA INT8 </option>
        <option value='gptj'>GPT-J</option>
        <option value='gptj_lora'>GPT-J LoRA</option>
        <option value='gptj_lora_int8'>GPT-J LoRA INT8</option>
        <option value='gpt2'>GPT-2</option>
        <option value='gpt2_lora'>GPT-2 LoRA </option>
        <option value='gpt2_lora_int8'>GPT-2 LoRA INT8 </option>
        <option value='distilgpt2'>DistilGPT-2 </option>
        <option value='distilgpt2_lora'>DistilGPT-2 LoRA</option>
        <option value='opt'>OPT</option>
        <option value='opt_lora'>OPT LoRA</option>
        <option value='opt_lora_int8'>OPT LoRA INT8</option>
        <option value='cerebras'>Cerebras </option>
        <option value='cerebras_lora'>Cerebras LoRA </option>
        <option value='cerebras_lora_int8'>Cerebras LoRA INT8 </option>
        <option value='galactica'>Galactica </option>
        <option value='galactica_lora'>Galactica LoRA </option>
        <option value='galactica_lora_int8'>Galactica LoRA INT8 </option>
        <option value='bloom'>BLOOM</option>
        <option value='bloom_lora'>BLOOM LoRA </option>
        <option value='bloom_lora_int8'>BLOOM LoRA INT8</option>
      </select>
      <CodeBlock
      className='row'
      showLineNumbers={false}
      language='python'
      children={`from xturing.datasets import ${instruction}Dataset
from xturing.models import BaseModel

dataset = ${instruction}Dataset('/path/to/dataset')
model = BaseModel.create('${code}')`} 
      
      
      />
    </div>
  )
}