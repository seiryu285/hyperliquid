describe('Risk Monitoring Dashboard', () => {
  beforeEach(() => {
    cy.visit('http://localhost:3000')
  })

  it('should display all risk metric cards', () => {
    cy.get('[data-testid="margin-buffer"]').should('exist')
    cy.get('[data-testid="volatility"]').should('exist')
    cy.get('[data-testid="liquidation-risk"]').should('exist')
    cy.get('[data-testid="value-at-risk"]').should('exist')
  })

  it('should establish WebSocket connection', () => {
    // Wait for WebSocket connection
    cy.window().then((win) => {
      expect(win.WebSocket).to.be.a('function')
    })

    // Check connection status indicator
    cy.get('[data-testid="connection-status"]')
      .should('have.text', 'Connected')
      .and('have.class', 'connected')
  })

  it('should update metrics in real-time', () => {
    // Get initial values
    cy.get('[data-testid="margin-buffer-value"]').then(($initial) => {
      const initialValue = $initial.text()

      // Wait for update
      cy.get('[data-testid="margin-buffer-value"]', { timeout: 10000 })
        .should(($updated) => {
          expect($updated.text()).not.to.eq(initialValue)
        })
    })
  })

  it('should display chart with historical data', () => {
    cy.get('canvas').should('exist')
    
    // Ensure chart has data points
    cy.window().then((win) => {
      const chart = win.document.querySelector('canvas').__chart__
      expect(chart.data.datasets[0].data.length).to.be.greaterThan(0)
    })
  })

  it('should handle connection loss gracefully', () => {
    // Simulate connection loss
    cy.window().then((win) => {
      win.WebSocket.prototype.close.call(
        win.document.querySelector('[data-testid="websocket"]').__ws
      )
    })

    // Check reconnection attempt
    cy.get('[data-testid="connection-status"]')
      .should('have.text', 'Reconnecting...')
      .and('have.class', 'reconnecting')

    // Wait for reconnection
    cy.get('[data-testid="connection-status"]', { timeout: 10000 })
      .should('have.text', 'Connected')
      .and('have.class', 'connected')
  })

  it('should display error states correctly', () => {
    // Check error state styling
    cy.get('[data-testid="margin-buffer"]').within(() => {
      cy.get('.value')
        .invoke('text')
        .then((text) => {
          const value = parseFloat(text)
          if (value < 0.2) {
            cy.get('.value').should('have.css', 'color', 'rgb(255, 0, 0)')
          }
        })
    })
  })

  it('should maintain performance under load', () => {
    // Monitor performance metrics
    cy.window().then((win) => {
      const t0 = performance.now()
      
      // Wait for multiple updates
      cy.get('[data-testid="margin-buffer-value"]')
        .should('not.eq', '0')
        .then(() => {
          const t1 = performance.now()
          const updateTime = t1 - t0
          expect(updateTime).to.be.lessThan(100) // 100ms threshold
        })
    })
  })
})
