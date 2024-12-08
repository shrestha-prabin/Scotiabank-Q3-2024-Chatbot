async function getMessage(query) {
  const res = await fetch('/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 'query': query })
  })
  const data = await res.json()
  console.log("🚀 ~ getMessage ~ data:", data)
  return data
}