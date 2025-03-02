import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Box, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  Divider,
  IconButton,
  ListItemButton,
  useMediaQuery,
  Avatar,
  Tooltip,
  Badge
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import SettingsIcon from '@mui/icons-material/Settings';
import MenuIcon from '@mui/icons-material/Menu';
import GitHubIcon from '@mui/icons-material/GitHub';
import NotificationsIcon from '@mui/icons-material/Notifications';
import LightModeIcon from '@mui/icons-material/LightMode';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import CloseIcon from '@mui/icons-material/Close';

import Dashboard from './pages/Dashboard';

// カスタムテーマの作成
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#6366f1', // Indigo
    },
    secondary: {
      main: '#ec4899', // Pink
    },
    background: {
      default: '#0f172a', // Slate 900
      paper: '#1e293b', // Slate 800
    },
    text: {
      primary: '#f1f5f9', // Slate 100
      secondary: '#94a3b8', // Slate 400
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
    },
    h6: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
          backgroundImage: 'none',
        },
      },
    },
  },
});

const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#6366f1', // Indigo
    },
    secondary: {
      main: '#ec4899', // Pink
    },
    background: {
      default: '#f8fafc', // Slate 50
      paper: '#ffffff', // White
    },
    text: {
      primary: '#0f172a', // Slate 900
      secondary: '#64748b', // Slate 500
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
    },
    h6: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
          backgroundImage: 'none',
        },
      },
    },
  },
});

const drawerWidth = 240;

function App() {
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [darkMode, setDarkMode] = React.useState(true);
  const isMobile = useMediaQuery('(max-width:900px)');

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleThemeToggle = () => {
    setDarkMode(!darkMode);
  };

  const drawer = (
    <div>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        p: 2 
      }}>
        <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 'bold' }}>
          HyperLiquid
        </Typography>
        {isMobile && (
          <IconButton onClick={handleDrawerToggle} size="small">
            <CloseIcon />
          </IconButton>
        )}
      </Box>
      <Divider />
      <List>
        <ListItem disablePadding>
          <ListItemButton component={Link} to="/" selected>
            <ListItemIcon>
              <DashboardIcon />
            </ListItemIcon>
            <ListItemText primary="ダッシュボード" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton component={Link} to="/trading">
            <ListItemIcon>
              <ShowChartIcon />
            </ListItemIcon>
            <ListItemText primary="トレーディング" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton component={Link} to="/portfolio">
            <ListItemIcon>
              <AccountBalanceWalletIcon />
            </ListItemIcon>
            <ListItemText primary="ポートフォリオ" />
          </ListItemButton>
        </ListItem>
      </List>
      <Divider />
      <List>
        <ListItem disablePadding>
          <ListItemButton component={Link} to="/settings">
            <ListItemIcon>
              <SettingsIcon />
            </ListItemIcon>
            <ListItemText primary="設定" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton component="a" href="https://github.com/seiryu285/hyperliquid" target="_blank">
            <ListItemIcon>
              <GitHubIcon />
            </ListItemIcon>
            <ListItemText primary="GitHub" />
          </ListItemButton>
        </ListItem>
      </List>
      <Box sx={{ mt: 'auto', p: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Version 1.0.0
        </Typography>
      </Box>
    </div>
  );

  return (
    <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <AppBar
            position="fixed"
            sx={{
              width: { md: `calc(100% - ${drawerWidth}px)` },
              ml: { md: `${drawerWidth}px` },
              bgcolor: darkMode ? 'background.paper' : 'background.paper',
            }}
          >
            <Toolbar>
              <IconButton
                color="inherit"
                aria-label="open drawer"
                edge="start"
                onClick={handleDrawerToggle}
                sx={{ mr: 2, display: { md: 'none' } }}
              >
                <MenuIcon />
              </IconButton>
              <Typography
                variant="h6"
                noWrap
                component="div"
                sx={{ display: { xs: 'none', sm: 'block' } }}
              >
                HyperLiquid Trading System
              </Typography>
              <Box sx={{ flexGrow: 1 }} />
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Tooltip title={darkMode ? "ライトモード" : "ダークモード"}>
                  <IconButton color="inherit" onClick={handleThemeToggle}>
                    {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="通知">
                  <IconButton color="inherit">
                    <Badge badgeContent={4} color="secondary">
                      <NotificationsIcon />
                    </Badge>
                  </IconButton>
                </Tooltip>
                <Tooltip title="プロフィール">
                  <Avatar
                    sx={{ 
                      width: 32, 
                      height: 32, 
                      bgcolor: 'primary.main',
                      cursor: 'pointer'
                    }}
                  >
                    U
                  </Avatar>
                </Tooltip>
              </Box>
            </Toolbar>
          </AppBar>
          <Box
            component="nav"
            sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
            aria-label="mailbox folders"
          >
            <Drawer
              variant="temporary"
              open={mobileOpen}
              onClose={handleDrawerToggle}
              ModalProps={{
                keepMounted: true, // Better open performance on mobile.
              }}
              sx={{
                display: { xs: 'block', md: 'none' },
                '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
              }}
            >
              {drawer}
            </Drawer>
            <Drawer
              variant="permanent"
              sx={{
                display: { xs: 'none', md: 'block' },
                '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
              }}
              open
            >
              {drawer}
            </Drawer>
          </Box>
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              width: { md: `calc(100% - ${drawerWidth}px)` },
              minHeight: '100vh',
              bgcolor: 'background.default',
            }}
          >
            <Toolbar />
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/trading" element={<ComingSoon title="トレーディング" />} />
              <Route path="/portfolio" element={<ComingSoon title="ポートフォリオ" />} />
              <Route path="/settings" element={<ComingSoon title="設定" />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

// 準備中ページのコンポーネント
function ComingSoon({ title }: { title: string }) {
  return (
    <Container sx={{ py: 8, textAlign: 'center' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {title}
      </Typography>
      <Typography variant="h6" color="text.secondary" paragraph>
        このページは現在開発中です。近日公開予定です。
      </Typography>
      <Box
        component="img"
        src="https://illustrations.popsy.co/amber/digital-nomad.svg"
        alt="Coming Soon"
        sx={{
          width: '100%',
          maxWidth: 400,
          mt: 4
        }}
      />
    </Container>
  );
}

export default App;
