document.addEventListener('DOMContentLoaded', function () {
  let embedOpen = false

  let styleBase = `position:fixed;width:396px;height:70vh; bottom:32px; right: 32px;z-index:2000; opacity:0; transition: all 300ms ease-in-out; pointer-events:none; border:1px solid #EBEBEB; border-radius: 16px; background:#fff; box-shadow: 0px 10px 16px rgba(0, 0, 0, 0.25);`
  let styleOpen = styleBase + 'opacity:1; pointer-events:auto;'

  let embedButtonStyle =
    'position:fixed;width:72px;height:72px; bottom:40px; right:40px; z-index:1000;background:#000;border-radius:12px; transition: transform 300ms ease-in-out, box-shadow 1500ms ease-in-out; cursor:pointer; background-color:#141718; box-shadow:0px 0px 8px 3px rgba(0,0,0, 0.2); '

  const vw = Math.max(
    document.documentElement.clientWidth || 0,
    window.innerWidth || 0
  )

  if (vw < 480) {
    styleBase = `position:fixed;width:calc(100% - 24px); height:calc(100% - 24px); bottom:12px; left: 12px;z-index:1000; opacity:0; transition: opacity 300ms ease-in-out; pointer-events:none; border:1px solid #EBEBEB; border-radius: 16px; box-shadow: 0px 16px 48px rgba(0, 0, 0, 0.07); background:#fff; box-shadow: 0px 10px 16px rgba(0, 0, 0, 0.25);`
    embedButtonStyle = embedButtonStyle += 'bottom:20px; right:20px;'
  }

  const embedButtonHover = embedButtonStyle + 'transform:scale(1.10);'
  const embedButtonPulse =
    embedButtonStyle + 'box-shadow:0px 0px 16px 3px rgba(0,0,0, 0.8);'

  //embed button
  const embedButton = document.createElement('div')
  embedButton.style.cssText = embedButtonStyle

  //iframe
  const elemIframe = document.createElement('iframe')
  elemIframe.src = window.xChatEmbed
  elemIframe.allow = 'clipboard-write;'
  elemIframe.style.cssText = styleBase

  const toggleEmbed = () => {
    embedOpen = !embedOpen
    if (embedOpen) {
      elemIframe.style.cssText = styleOpen
    } else {
      elemIframe.style.cssText = styleBase
    }
  }

  const embedHover = () => {
    embedButton.style.cssText = embedButtonHover
  }

  const embedDeafult = () => {
    embedButton.style.cssText = embedButtonStyle
  }

  const pulseIn = () => {
    embedButton.style.cssText = embedButtonPulse
    setTimeout(() => pulseOut(), 1500)
  }

  const pulseOut = () => {
    embedButton.style.cssText = embedButtonStyle
    setTimeout(() => pulseIn(), 1500)
  }

  embedButton.addEventListener('mouseover', embedHover)
  embedButton.addEventListener('mouseout', embedDeafult)
  embedButton.addEventListener('click', () => {
    toggleEmbed()
  })

  //add to DOM
  document.body.appendChild(embedButton)
  document.body.appendChild(elemIframe)
  pulseIn()
})
