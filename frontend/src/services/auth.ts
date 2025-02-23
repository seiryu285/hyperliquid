/**
 * Authentication service for the HyperLiquid trading system.
 */

import axios from 'axios';
import { User } from '../types/user';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class AuthService {
    async login(email: string, password: string) {
        const response = await axios.post(`${API_URL}/auth/login`, {
            email,
            password
        });
        if (response.data.access_token) {
            localStorage.setItem('user', JSON.stringify(response.data));
        }
        return response.data;
    }

    async register(username: string, email: string, password: string) {
        return await axios.post(`${API_URL}/auth/register`, {
            username,
            email,
            password
        });
    }

    logout() {
        localStorage.removeItem('user');
    }

    getCurrentUser() {
        const userStr = localStorage.getItem('user');
        if (userStr) {
            return JSON.parse(userStr);
        }
        return null;
    }

    async requestPasswordReset(email: string) {
        return await axios.post(`${API_URL}/security/password-reset/request`, {
            email
        });
    }

    async confirmPasswordReset(token: string, newPassword: string) {
        return await axios.post(`${API_URL}/security/password-reset/confirm`, {
            token,
            new_password: newPassword
        });
    }

    async setup2FA() {
        const response = await axios.post(
            `${API_URL}/security/2fa/setup`,
            {},
            {
                headers: this.authHeader()
            }
        );
        return response.data;
    }

    async verify2FA(token: string) {
        const response = await axios.post(
            `${API_URL}/security/2fa/verify`,
            { token },
            {
                headers: this.authHeader()
            }
        );
        if (response.data.access_token) {
            const user = this.getCurrentUser();
            user.access_token = response.data.access_token;
            localStorage.setItem('user', JSON.stringify(user));
        }
        return response.data;
    }

    async disable2FA(token: string) {
        return await axios.post(
            `${API_URL}/security/2fa/disable`,
            { token },
            {
                headers: this.authHeader()
            }
        );
    }

    private authHeader() {
        const user = this.getCurrentUser();
        if (user && user.access_token) {
            return { Authorization: `Bearer ${user.access_token}` };
        }
        return {};
    }
}

export const authService = new AuthService();
