async function getMessage(source, query) {
  const res = await fetch('/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, query })
  })
  const data = await res.json()
  console.log("🚀 ~ getMessage ~ data:", data)
  return data
}