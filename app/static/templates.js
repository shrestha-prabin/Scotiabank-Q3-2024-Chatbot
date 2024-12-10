const messageTemplates = {
  user: (text) => `
    <div class="flex justify-end">
      <div class="flex bg-white border rounded-full p-4 text-end max-w-[80%]">
        ${text}
      </div>
    </div>
  `,
  bot: (text) => `
    <div class="flex flex-row items-start gap-4 mb-4">
      <div class="h-12 w-12 rounded-full bg-white grid place-items-center">
        ${scotiaLogo}
      </div>
      <div class="flex-1 py-2">
        ${text}
      </div>
    </div>
  `
}