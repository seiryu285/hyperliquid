/// <reference types="cypress" />

import './commands'

// Cypressグローバル設定をここに追加できます
Cypress.on('uncaught:exception', () => {
  return false
})

export {}
