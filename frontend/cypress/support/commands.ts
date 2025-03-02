/// <reference types="cypress" />

// ***********************************************
// This example commands.ts shows you how to
// create various custom commands and overwrite
// existing commands.
//
// For more comprehensive examples of custom
// commands please read more here:
// https://on.cypress.io/custom-commands
// ***********************************************

declare global {
  namespace Cypress {
    interface Chainable {
      getByTestId(testId: string): Chainable<JQuery<HTMLElement>>
      checkWebSocketConnection(): Chainable<boolean>
    }
  }
}

Cypress.Commands.add('getByTestId', (testId: string) => {
  return cy.get(`[data-testid="${testId}"]`)
})

Cypress.Commands.add('checkWebSocketConnection', () => {
  return cy.window().then((win) => {
    return new Promise<boolean>((resolve) => {
      const ws = new win.WebSocket('ws://localhost:8080')
      ws.onopen = () => {
        ws.close()
        resolve(true)
      }
      ws.onerror = () => {
        resolve(false)
      }
    })
  })
})

export {}
