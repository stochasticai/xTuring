import React, { useEffect, useState } from 'react'
import clsx from 'clsx'
import MDXContent from '@theme/MDXContent'
import CodeBlock from '@theme/CodeBlock'

const trainingTechniques = {
  base: 'Base',
  lora: 'LoRa',
  lora_int8: 'LoRA INT8',
  int8: 'INT8',
}

const modelList = {
  bloom: 'BLOOM',
  cerebras: 'Cerebras',
  distilgpt2: 'DistilGPT-2',
  galactica: 'Galactica',
  gptj: 'GPT-J', 
  gpt2: 'GPT-2',
  llama: 'LLaMA',
  llama2: 'LLaMA 2',
  opt: 'OPT',
}

export default function Test(
  {instruction}
) {
  // const [code, setCode] = useState('llama');
  const [code, setCode] = useState({
    model: '',
    technique: 'base',
  })

  let finalKey = ''
  if (code.technique === 'base') {
    finalKey = `${code.model}`
  } else {
    finalKey = `${code.model}_${code.technique}`
  }
  
  useEffect(() => {
    setCode({
      model: 'bloom',
      technique: 'base'
    });
  }, []);

  return (
    <div className={clsx('col')}>
      <label htmlFor='model'>Choose a model: </label>
      <select
        style={{ padding: '8px 16px', borderRadius: '8px', marginRight:'5px' }}
        name='model'
        id='model'
        onChange={(e) =>
          setCode((prev) => ({
            ...prev,
            model: e.target.value,
          }))
        }
      >
        {Object.keys(modelList).map((key) => (
          <option value={key}>{modelList[key]}</option>
        ))}
      </select>

      <label htmlFor='tech'>Choose version: </label>
      <select
        style={{ padding: '8px 16px', borderRadius: '8px', marginLeft:'5px' }}
        name='tech'
        id='tech'
        onChange={(e) =>
          setCode((prev) => ({
            ...prev,
            technique: e.target.value,
          }))
        }
      >
        {Object.keys(trainingTechniques).map((key) => (
          <option value={key}>{trainingTechniques[key]}</option>
        ))}
      </select>

      <CodeBlock
      className='row'
      showLineNumbers={false}
      language='python'
      children={`from xturing.datasets import ${instruction}Dataset
from xturing.models import BaseModel

dataset = ${instruction}Dataset('/path/to/dataset')
model = BaseModel.create('${finalKey}')`} 
      />
    </div>
  )
}