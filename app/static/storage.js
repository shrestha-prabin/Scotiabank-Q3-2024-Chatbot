let db = JSON.parse(localStorage.getItem('storage')) ?? []

function save() {
  localStorage.setItem('storage', JSON.stringify(db))
}

const DB = {
  getAll: () => {
    console.log("🚀 ~ db:", db)
    return db
  },
  add: (role, message) => {
    db.push({ role, message })
    save()
    console.log("🚀 ~ add")
  },
  clear: () => {
    db = []
    save()
    console.log("🚀 ~ clear")
  }
}

