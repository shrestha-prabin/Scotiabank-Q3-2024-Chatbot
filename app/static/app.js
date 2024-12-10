const chatContent = document.getElementById("chat-content");
const clearAllButton = document.getElementById('clear-all-btn')

function addChatContent(role, content) {
  const newContent = messageTemplates[role](content)
  chatContent.innerHTML += newContent
}

function scrollToBottom(behavior) {
  const el = document.getElementById('bottom-content')
  el.scrollIntoView({ behavior: behavior || 'smooth' })
}

async function handleSubmit(value) {
  console.log("🚀 ~ handleSubmit ~ value:", value)

  const inputEl = document.getElementsByName('query')[0]
  const query = value || inputEl.value
  inputEl.value = ''

  if (query) {
    try {
      addChatContent('user', query)
      clearAllButton.style.opacity = 1
      const res = await getMessage(modelPreference, query)
      addChatContent('bot', res)

      DB.add('user', query)
      DB.add('bot', res)

      scrollToBottom()
    } catch (error) {
      console.log("🚀 ~ submitButton.addEventListener ~ error:", error)
      alert('Something went wrong')
    }
  }
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    handleSubmit()
  }
});

function init() {
  const defaultData = DB.getAll()

  if (defaultData.length == 0) {
    clearAllButton.style.opacity = 0
  } else {
    clearAllButton.style.opacity = 1
  }

  defaultData.forEach(item => {
    addChatContent(item.role, item.message)
  });

  clearAllButton.addEventListener('click', () => {
    DB.clear()
    chatContent.innerHTML = ''
    clearAllButton.style.opacity = 0
  })
}

let modelPreference = 'text'

const toggleTextButton = document.getElementById('toggle-text-btn')
const toggleTableButton = document.getElementById('toggle-table-btn')

toggleTextButton.addEventListener('click', () => {
  modelPreference = 'text'
  toggleTextButton.setAttribute('data-active', 'true');
  toggleTableButton.removeAttribute('data-active');
})

toggleTableButton.addEventListener('click', () => {
  modelPreference = 'table'
  toggleTableButton.setAttribute('data-active', 'true');
  toggleTextButton.removeAttribute('data-active');
})

init()
